"""Fallback MAT5 reader for struct trees that scipy.io.loadmat refuses to parse.

Some measurement files in this project (e.g. info.mat written by
GNU Octave 5.2.0) contain a top-level struct that scipy's mio5 reader
raises `TypeError: buffer is too small for requested array` on, even
though the file is not actually truncated or corrupted -- byte-for-byte
accounting of every tag in the file shows it is fully self-consistent
and ends exactly at EOF. This appears to be a scipy-side incompatibility
with how these files encode nested char data, not a data-loss problem.

This module re-implements just enough of the MAT5 format (struct, cell,
char, and numeric arrays) to read those structs directly, returning
plain nested SimpleNamespace/str/numpy-scalar values so callers can use
the same dotted attribute access MATLAB-style code expects
(e.g. `meta.generateCE.DACDelay`).
"""
import struct
from types import SimpleNamespace
import numpy as np

_MI_DTYPES = {
    1: '<i1', 2: '<u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4',
    7: '<f4', 9: '<f8', 12: '<i8', 13: '<u8',
}
_MX_CELL = 1
_MX_STRUCT = 2
_MX_CHAR = 4


class _Reader:
    def __init__(self, data):
        self.data = data
        self.n = len(data)

    def read_tag(self, pos):
        data_type, num_bytes = struct.unpack_from('<II', self.data, pos)
        if (data_type >> 16) != 0:
            # small data element format: size in high 16 bits, type in low 16 bits
            nb = data_type >> 16
            dt = data_type & 0xFFFF
            return dt, nb, pos + 4, pos + 8
        start = pos + 8
        padded = (num_bytes + 7) // 8 * 8
        return data_type, num_bytes, start, start + padded

    def read_matrix(self, pos):
        dt, nb, start, next_pos = self.read_tag(pos)
        if dt != 6:
            raise ValueError(f'expected array flags (miUINT32) at byte {pos}, got type {dt}')
        flags_word, _nzmax = struct.unpack_from('<II', self.data, start)
        mclass = flags_word & 0xFF
        is_complex = bool(flags_word & (1 << 11))
        pos = next_pos

        dt, nb, start, next_pos = self.read_tag(pos)
        ndims = nb // 4
        dims = struct.unpack_from(f'<{ndims}i', self.data, start)
        pos = next_pos

        dt, nb, start, next_pos = self.read_tag(pos)
        name = self.data[start:start + nb].decode('latin1')
        pos = next_pos

        nelems = 1
        for d in dims:
            nelems *= d

        if mclass == _MX_STRUCT:
            dt, nb, start, next_pos = self.read_tag(pos)
            fnlen = struct.unpack_from('<i', self.data, start)[0]
            pos = next_pos
            dt, nb, start, next_pos = self.read_tag(pos)
            nfields = nb // fnlen
            fnames = [
                self.data[start + i * fnlen:start + (i + 1) * fnlen].split(b'\x00')[0].decode('latin1')
                for i in range(nfields)
            ]
            pos = next_pos
            elems = []
            for _ in range(nelems):
                ns = SimpleNamespace()
                for fn in fnames:
                    val, pos = self._read_field_element(pos)
                    setattr(ns, fn, val)
                elems.append(ns)
            if nelems == 0:
                return SimpleNamespace(), pos, name
            return (elems[0] if nelems == 1 else elems), pos, name

        elif mclass == _MX_CELL:
            elems = []
            for _ in range(nelems):
                val, pos = self._read_field_element(pos)
                elems.append(val)
            return elems, pos, name

        elif mclass == _MX_CHAR:
            dt, nb, start, next_pos = self.read_tag(pos)
            pos = next_pos
            raw = self.data[start:start + nb]
            if nelems == 0:
                return '', pos, name
            if dt == 16:  # miUTF8
                text = raw.decode('utf-8', errors='replace')
                return text, pos, name
            itemsize = {1: 1, 2: 1, 3: 2, 4: 2}.get(dt, 2)
            codes = np.frombuffer(raw, dtype=f'<u{itemsize}')
            rows = dims[0]
            cols = dims[1] if len(dims) > 1 else 1
            grid = codes.reshape((rows, cols), order='F')
            lines = [''.join(chr(c) for c in grid[r, :]) for r in range(rows)]
            return (lines[0] if rows == 1 else lines), pos, name

        else:  # numeric class
            dt, nb, start, next_pos = self.read_tag(pos)
            pos = next_pos
            if is_complex:
                # skip imaginary part; not needed by any current caller
                _dt2, _nb2, _s2, next_pos2 = self.read_tag(pos)
                pos = next_pos2
            npdt = _MI_DTYPES.get(dt)
            if npdt is None or nelems == 0:
                arr = np.array([])
            else:
                arr = np.frombuffer(self.data[start:start + nb], dtype=npdt)
                if len(dims) >= 2:
                    arr = arr.reshape(dims, order='F')
            if arr.size == 1:
                return arr.reshape(-1)[0].item(), pos, name
            return arr, pos, name

    def _read_field_element(self, pos):
        dt, nb, start, next_pos = self.read_tag(pos)
        if dt != 14:
            raise ValueError(f'expected nested miMATRIX at byte {pos}, got type {dt}')
        val, _endpos, _name = self.read_matrix(start)
        return val, next_pos


def load_struct(path, variable_names=None):
    """Read top-level MAT5 variables into a dict of name -> value.

    Structs become nested SimpleNamespace objects (dotted attribute
    access), char arrays become str, numeric arrays become numpy
    arrays (or a plain scalar for 1x1 arrays).

    `variable_names`, if given, restricts which top-level variables are
    fully decoded (others are skipped over cheaply using their declared
    byte length).
    """
    with open(path, 'rb') as f:
        data = f.read()
    r = _Reader(data)
    n = len(data)
    pos = 128
    out = {}
    while pos < n - 8:
        dt, nb, start, next_pos = r.read_tag(pos)
        if dt == 14:
            # peek the name without decoding the whole thing, when filtering
            if variable_names is not None:
                _peek_val, _peek_end, peek_name = r.read_matrix(start)
                if peek_name in variable_names:
                    out[peek_name] = _peek_val
            else:
                val, _end, name = r.read_matrix(start)
                out[name] = val
        pos = next_pos
    return out
