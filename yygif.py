from enum import Enum
import io
import os
from struct import unpack
from typing import Callable, NamedTuple, Optional

import numpy as np


class Error(RuntimeError):
    pass


class DecoderError(Error):
    pass


def _read_bytes(fp: io.BytesIO, size: int, name: str) -> bytes:
    buffer = fp.read(size)
    if len(buffer) < size:
        raise DecoderError(f'EOF encountered when reading {name}')
    return buffer


def _read_color_table(fp: io.BytesIO, size: int, name: str) -> np.ndarray:
    n_colors = 2 ** (size + 1)
    buffer = _read_bytes(fp, n_colors * 3, name)
    return np.frombuffer(buffer, np.uint8).reshape((n_colors, 3))


def _read_data_sub_blocks(fp: io.BytesIO, name: str) -> bytes:
    blocks = []
    while True:
        block_size = _read_bytes(fp, 1, name)[0]
        if block_size == 0:
            return b''.join(blocks)
        blocks.append(_read_bytes(fp, block_size, name))


def _skip_data_sub_blocks(fp: io.BytesIO, name: str) -> None:
    while True:
        block_size = _read_bytes(fp, 1, name)[0]
        if block_size == 0:
            return
        fp.seek(block_size, os.SEEK_CUR)


def _get_area_view(
        array: np.ndarray,
        size: tuple[int, ...],
        crop_position: tuple[int, int],
        crop_size: tuple[int, int],
) -> np.ndarray:
    view = array.reshape(size)
    return view[crop_position[0]:crop_position[0] + crop_size[0],
                crop_position[1]:crop_position[1] + crop_size[1]]


def _lzw_decode(minimum_code_size: int, data: bytes, out: np.ndarray) -> None:
    # TODO
    pass


class Header(NamedTuple):
    version: str

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'Header':
        signature = _read_bytes(fp, 3, 'header')
        if signature != b'GIF':
            raise DecoderError('Not a GIF image')
        version = _read_bytes(fp, 3, 'header')
        if version not in (b'87a', b'89a'):
            raise DecoderError(f'Invalid GIF version "{version.decode()}"')
        return cls(version.decode())


class LogicalScreen(NamedTuple):
    size: tuple[int, int]
    color_resolution: int
    sort_flag: Optional[bool]
    background_color_index: Optional[np.ndarray]
    pixel_aspect_ratio: int
    color_table: Optional[np.ndarray]

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'LogicalScreen':
        buffer = _read_bytes(fp, 7, 'logical screen descriptor')
        (
            width, height,
            packed_fields,
            background_color_index,
            pixel_aspect_ratio,
        ) = unpack('<HHBBB', buffer)
        if height == 0 or width == 0:
            raise DecoderError('Invalid logical screen size '
                               f'({height}, {width})')
        global_color_table_flag = bool(packed_fields >> 7)
        color_resolution = (packed_fields >> 4) & 0b111
        sort_flag = bool((packed_fields >> 3) & 1)
        color_table_size = packed_fields & 0b111
        if global_color_table_flag:
            color_table = \
                _read_color_table(fp, color_table_size, 'global color table')
            n_colors = 2 ** (color_table_size + 1)
            if background_color_index >= n_colors:
                raise DecoderError(
                    f'Background color index ({background_color_index}) '
                    f'out of bounds ({n_colors})'
                )
        else:
            color_table = None
        return cls(
            (height, width),
            color_resolution,
            sort_flag if global_color_table_flag else None,
            background_color_index if global_color_table_flag else None,
            pixel_aspect_ratio,
            color_table,
        )


class Image(NamedTuple):
    position: tuple[int, int]
    size: tuple[int, int]
    interlace_flag: bool
    sort_flag: Optional[bool]
    color_table: Optional[np.ndarray]
    minimum_code_size: int
    data: bytes

    @classmethod
    def read(cls, fp: io.BytesIO):
        buffer = _read_bytes(fp, 9, 'image descriptor')
        (
            left_position, top_position,
            width, height,
            packed_fields,
        ) = unpack('<HHHHB', buffer)
        local_color_table_flag = bool(packed_fields >> 7)
        interlace_flag = bool((packed_fields >> 6) & 1)
        sort_flag = bool((packed_fields >> 5) & 1)
        color_table_size = packed_fields & 0b111
        if local_color_table_flag:
            color_table = \
                _read_color_table(fp, color_table_size, 'local color table')
        else:
            color_table = None
        minimum_code_size = _read_bytes(fp, 1, 'table based image data')[0]
        if minimum_code_size == 0:
            raise DecoderError('Minimum code size cannot be zero')
        data = _read_data_sub_blocks(fp, 'table based image data')
        return cls(
            (top_position, left_position),
            (height, width),
            interlace_flag,
            sort_flag if local_color_table_flag else None,
            color_table,
            minimum_code_size,
            data,
        )


class DisposalMethod(Enum):
    NOT_SPECIFIED = 0
    DO_NOT_DISPOSE = 1
    RESTORE_TO_BACKGROUND = 2
    RESTORE_TO_PREVIOUS = 3


class GraphicControl(NamedTuple):
    disposal_method: DisposalMethod
    user_input_flag: bool
    delay_time: int
    transparent_color_index: Optional[int]

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'GraphicControl':
        buffer = _read_data_sub_blocks(fp, 'graphic control extension')
        if len(buffer) != 4:
            raise DecoderError('graphic control extension must have size 4, '
                               f'got {len(buffer)}')
        packed_fields, delay_time, transparent_color_index = \
            unpack('<BHB', buffer)
        disposal_method = (packed_fields >> 2) & 0b111
        if not 0 <= disposal_method <= 3:
            raise DecoderError('disposal method must be between 0 and 3, '
                               f'got {disposal_method}')
        user_input_flag = bool((packed_fields >> 1) & 1)
        transparent_color_flag = bool(packed_fields & 1)
        return cls(
            DisposalMethod(disposal_method),
            user_input_flag,
            delay_time,
            transparent_color_index if transparent_color_flag else None,
        )


class PlainText(NamedTuple):
    position: tuple[int, int]
    size: tuple[int, int]
    cell_size: tuple[int, int]
    foreground_color_index: int
    background_color_index: int
    data: str

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'PlainText':
        buffer = _read_data_sub_blocks(fp, 'plain text extension')
        if len(buffer) != 12:
            raise DecoderError('plain text extension must have size 4, '
                               f'got {len(buffer)}')
        (
            left_position, top_position,
            width, height,
            cell_width, cell_height,
            foreground_color_index, background_color_index
        ) = unpack('<HHHHBBBB', buffer)
        data = _read_data_sub_blocks(fp, 'plain text extension').decode()
        return cls(
            (top_position, left_position),
            (height, width),
            (cell_height, cell_width),
            foreground_color_index,
            background_color_index,
            data,
        )

    @staticmethod
    def skip(fp: io.BytesIO) -> None:
        _skip_data_sub_blocks(fp, 'plain text extension')
        _skip_data_sub_blocks(fp, 'plain text extension')


class Comment(NamedTuple):
    data: str

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'Comment':
        data = _read_data_sub_blocks(fp, 'comment extension').decode()
        return cls(data)

    @staticmethod
    def skip(fp: io.BytesIO) -> None:
        _skip_data_sub_blocks(fp, 'comment extension')


class ApplicationExtension(NamedTuple):
    identifier: str
    authentication_code: str
    data: bytes

    @classmethod
    def read(cls, fp: io.BytesIO) -> 'ApplicationExtension':
        buffer = _read_data_sub_blocks(fp, 'application extension')
        if len(buffer) != 11:
            raise DecoderError('Application extension must have size 11, '
                               f'got {len(buffer)}')
        identifier = buffer[:8].decode()
        authentication_code = buffer[8:].decode()
        data = _read_data_sub_blocks(fp, 'application extension')
        return cls(identifier, authentication_code, data)


class _DisposalInfo(NamedTuple):
    method: DisposalMethod
    position: tuple[int, int]
    size: tuple[int, int]


class Decoder:
    def __init__(self):
        self._fp: Optional[io.BytesIO] = None
        self._version: Optional[str] = None
        self._logical_screen: Optional[LogicalScreen] = None
        self._graphic: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._restore_graphic: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._index_array: Optional[np.ndarray] = None
        self._interlaced_index_array: Optional[np.ndarray] = None
        self._graphic_control: Optional[GraphicControl] = None
        self._disposal_info: Optional[_DisposalInfo] = None
        self._plain_text_hook: Optional[Callable[[Decoder, PlainText], None]] \
            = None
        self._comment_hook: Optional[Callable[[Decoder, str], None]] = None
        self._application_hooks: dict[
            tuple[str, str],
            Callable[[Decoder, bytes], None],
        ] = {}

    def close(self) -> None:
        if self._fp is None:
            return
        self._fp.close()
        self._version = None
        self._logical_screen = None
        self._graphic = None
        self._restore_graphic = None
        self._index_array = None
        self._interlaced_index_array = None
        self._graphic_control = None
        self._disposal_info = None

    def open(self, fp: io.BytesIO) -> None:
        self.close()
        self._fp = fp
        self._version = Header.read(fp).version
        self._logical_screen = LogicalScreen.read(fp)
        n_pixels = self._logical_screen.size[0] * self._logical_screen.size[1]
        self._graphic = \
            np.empty(n_pixels * 3, np.uint8), np.empty(n_pixels, np.bool_)
        self._restore_graphic = \
            np.empty(n_pixels * 3, np.uint8), np.empty(n_pixels, np.bool_)
        self._index_array = np.empty(n_pixels, np.uint8)
        self._interlaced_index_array = np.empty(n_pixels, np.uint8)

    def read(self) -> bool:
        if self._disposal_info is not None:
            self._dispose()
        while True:
            separator = _read_bytes(self._fp, 1, 'separator')[0]
            if separator == 0x3b:
                return False
            if separator == 0x2c:
                self._read_image()
                if self._graphic_control is None or \
                        self._graphic_control.delay_time == 0:
                    continue
                return True
            if separator != 0x21:
                raise DecoderError(f'Invalid separator {separator:#02x}')
            label = _read_bytes(self._fp, 1, 'extension label')[0]
            if label == 0x01:
                self._read_plain_text()
                if self._graphic_control is None or \
                        self._graphic_control.delay_time == 0:
                    continue
                return True
            elif label == 0xf9:
                self._read_graphic_control()
            elif label == 0xfe:
                self._read_comment()
            elif label == 0xff:
                self._read_application_extension()
            else:
                raise DecoderError(f'Invalid extension label {label:#02x}')

    def _dispose(self) -> None:
        rgb_buffer = _get_area_view(
            self._graphic[0],
            self._logical_screen.size + (3,),
            self._disposal_info.position,
            self._disposal_info.size,
        )
        opacity_buffer = _get_area_view(
            self._graphic[1],
            self._logical_screen.size,
            self._disposal_info.position,
            self._disposal_info.size,
        )
        if self._disposal_info.method == DisposalMethod.RESTORE_TO_BACKGROUND:
            if self._logical_screen.color_table is None:
                rgb_buffer.fill(0)
                opacity_buffer.fill(0)
            else:
                background_color = self._logical_screen.color_table[
                    self._logical_screen.background_color_index]
                np.copyto(rgb_buffer, background_color)
                opacity_buffer.fill(1)
        elif self._disposal_info.method == DisposalMethod.RESTORE_TO_PREVIOUS:
            previous_rgb_buffer = _get_area_view(
                self._restore_graphic[0],
                self._logical_screen.size + (3,),
                self._disposal_info.position,
                self._disposal_info.size,
            )
            previous_opacity_buffer = _get_area_view(
                self._restore_graphic[1],
                self._logical_screen.size,
                self._disposal_info.position,
                self._disposal_info.size,
            )
            np.copyto(rgb_buffer, previous_rgb_buffer)
            np.copyto(opacity_buffer, previous_opacity_buffer)
        self._graphic_control = None
        self._disposal_info = None

    def _read_image(self) -> None:
        image = Image.read(self._fp)
        if image.interlace_flag:
            _lzw_decode(image.minimum_code_size, image.data,
                        self._interlaced_index_array)
            # TODO
        else:
            _lzw_decode(image.minimum_code_size, image.data,
                        self._index_array)
        if image.color_table is None:
            color_table = self._logical_screen.color_table
        else:
            color_table = image.color_table
        # TODO

    def _read_application_extension(self) -> None:
        extension = ApplicationExtension.read(self._fp)
        try:
            hook = self._application_hooks[
                (extension.identifier, extension.authentication_code)]
        except KeyError:
            pass
        else:
            hook(self, extension.data)

    def _read_comment(self) -> None:
        if self._comment_hook is None:
            Comment.skip(self._fp)
        else:
            self._comment_hook(self, Comment.read(self._fp).data)

    def _read_graphic_control(self) -> None:
        if self._graphic_control is not None:
            raise DecoderError('More than one graphic control extension '
                               'before a graphic rendering block')
        self._graphic_control = GraphicControl.read(self._fp)

    def _read_plain_text(self) -> None:
        if self._plain_text_hook is None:
            PlainText.skip(self._fp)
            self._disposal_info = _DisposalInfo(DisposalMethod.DO_NOT_DISPOSE,
                                                (0, 0), (0, 0))
        else:
            self._plain_text_hook(self, PlainText.read(self._fp))
