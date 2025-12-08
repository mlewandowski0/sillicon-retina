import cv2
import platform
import numpy as np


class SmartImshow(object):
    ESCAPE_KEY = 27
    NO_SELECTION = -1
    SELECTION_ON_GOING = 0
    SELECTION_IS_VALID = 1

    def __init__(self, window_name):
        
        # Publique
        self.input_image = None
        self.overlay_image = None

        # Private
        self._window_name = window_name
        self._bits_per_pixel = 0
        self._last_key = 0
        self._selection_state = self.NO_SELECTION
        self._hide_overlay = False
        self._add_grid = False
        self._grid_values = [[4 * j + i for i in range(4)] for j in range(4)]
        self._grid_colors = [[(0, 0, 0)] * 4] * 4

        # Protected
        self.__cursor_position = (0, 0)
        self.__button_down_position = self.__cursor_position
        self.__button_up_position = self.__cursor_position
        self.__zoom_center = (0, 0)
        self.__zoom_start_roi = (0, 0)
        self.__zoom_end_roi = (0, 0)
        self.__zoom_factor = 1

        # Set window and calls
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_call_back)

    @property
    def _image_width(self):
        if self.input_image is None:
            return 0
        return self.input_image.shape[1]

    @property
    def _image_height(self):
        if self.input_image is None:
            return 0
        return self.input_image.shape[0]

    @property
    def _top_left_corner(self):
        return (min(self.__button_down_position[0], self.__button_up_position[0]),
                min(self.__button_down_position[1], self.__button_up_position[1]))

    @property
    def _bottom_right_corner(self):
        return (max(self.__button_down_position[0], self.__button_up_position[0]),
                max(self.__button_down_position[1], self.__button_up_position[1]))

    @property
    def _cursor_position(self):
        return self.__cursor_position

    @_cursor_position.setter
    def _cursor_position(self, cursor_position):
        x, y = cursor_position
        x = min(max(0, x), self._image_width - 1)
        y = min(max(0, y), self._image_height - 1)
        self.__cursor_position = (x, y)

    @property
    def _overlay_color(self):
        if self._bits_per_pixel == 8:
            overlay_color = 255
        else:
            overlay_color = 65535
        if self._add_grid:
            if self._bits_per_pixel == 8:
                overlay_color = (255, 255, 255)
            else:
                overlay_color = (1023, 1023, 1023)
        return overlay_color


    def _check_selection(self):
        selection_width = self._bottom_right_corner[0] - self._top_left_corner[0]
        selection_height = self._bottom_right_corner[1] - self._top_left_corner[1]
        if selection_width and selection_height:
            self._selection_state = self.SELECTION_IS_VALID
        else:
            self._selection_state = self.NO_SELECTION

    def _compute_new_zoom_roi(self, new_zoom, cursor_position):
        self.__zoom_factor += new_zoom
        if self.__zoom_factor < 1:
            self.__zoom_factor = 1

    def _mouse_call_back(self, event, x, y, flags, param):
        # Store cursor
        self._cursor_position = (x, y)
        # Clicks to draw corners
        if event == cv2.EVENT_RBUTTONDOWN:
            self._selection_state = self.NO_SELECTION
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._selection_state = self.SELECTION_ON_GOING
            self.__button_down_position = self._cursor_position
            self.__button_up_position = self._cursor_position
        elif event == cv2.EVENT_LBUTTONUP:
            self.__button_up_position = self._cursor_position
            self._check_selection()
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom = int(flags/7864320)
            self._compute_new_zoom_roi(zoom, self._cursor_position)
        # Draw the overlay
        system = platform.system()
        if system == "Windows":
            self._generate_overlay_and_show_image()

    def _print_value(self, x, y, text, color, scale: float = 1):
        self.overlay_image = cv2.putText(self.overlay_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color)

    def _print_pixel_value(self):
        if self.input_image is not None:
            pixel_value = self.input_image[self._cursor_position[1], self._cursor_position[0]]
            self._print_value(50, 50, str(self._cursor_position) + ": " + str(pixel_value), self._overlay_color)

    def _draw_selection_box(self):
        if self._selection_state == self.SELECTION_IS_VALID:
            cv2.rectangle(self.overlay_image, self._top_left_corner, self._bottom_right_corner, self._overlay_color, 2, 8)
        elif self._selection_state == self.SELECTION_ON_GOING:
            cv2.rectangle(self.overlay_image, self._top_left_corner, self._cursor_position, 0, 2, 8)

    def _compute_image_stats(self):
        image_mean = np.round(np.mean(self.input_image), 2)
        image_std = np.round(np.std(self.input_image), 2)
        self._print_value(50, 100, "mean: " + str(image_mean), self._overlay_color)
        self._print_value(50, 150, "std: " + str(image_std), self._overlay_color)

    def _compute_selection_stats(self):
        if self._selection_state == self.SELECTION_IS_VALID:
            image_extract = self.input_image[
                            self._top_left_corner[1]:self._bottom_right_corner[1],
                            self._top_left_corner[0]:self._bottom_right_corner[0]
                            ]
            selection_mean = np.round(np.mean(image_extract), 2)
            selection_std = np.round(np.std(image_extract), 2)
            self._print_value(self._bottom_right_corner[0] + 5, self._top_left_corner[1], "mean: " + str(selection_mean), self._overlay_color)
            self._print_value(self._bottom_right_corner[0] + 5, self._top_left_corner[1] + 50, "std: " + str(selection_std), self._overlay_color)

    def add_grid(self):
        # Calculate the size of each cell
        cell_width = self._image_width // 4
        cell_height = self._image_height // 4
        self.overlay_image = cv2.cvtColor(self.overlay_image, cv2.COLOR_GRAY2RGB)
        # Draw the grid and text
        for row in range(4):
            for col in range(4):
                # Calculate the position of the cell
                top_left = (col * cell_width, row * cell_height)
                bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
                thickness = 2

                # Draw the rectangle
                color = self._grid_colors[row][col]
                msb_shift = 1 << (16 - self._bits_per_pixel)
                if self._bits_per_pixel in [10, 12, 14]:
                    color = (color[0] * msb_shift, color[1] * msb_shift, color[2] * msb_shift)
                cv2.rectangle(self.overlay_image, top_left, bottom_right, color=color, thickness=thickness)

                # Calculate the position of the text and draw it
                text_to_draw = f"{self._grid_values[row][col]}"
                text_position = [top_left[0] + 10, top_left[1] + 30]
                for line_to_draw in text_to_draw.split("\n"):
                    self._print_value(text_position[0], text_position[1], line_to_draw, color, scale=0.5)
                    (text_width, text_height), baseline = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                                          thickness)
                    text_position[1] += text_height + 10


    def _generate_overlay(self):
        if self.input_image is not None:
            # Copy input
            self.overlay_image = np.copy(self.input_image)

            # Zoom in image
            if self.__zoom_factor > 1:
                self.overlay_image = self.overlay_image[::self.__zoom_factor, ::self.__zoom_factor].repeat(self.__zoom_factor, axis=0).repeat(self.__zoom_factor, axis=1)

            # Adjust MSB for display
            if self._bits_per_pixel in [10, 12, 14]:
                self.overlay_image = self.overlay_image * (1 << (16 - self._bits_per_pixel))

            if not self._hide_overlay:
                # Print out everything
                self._print_pixel_value()
                self._compute_image_stats()
                self._draw_selection_box()
                self._compute_selection_stats()

    def _show_image(self):
        if self.overlay_image is not None and self.overlay_image.size > 0:
            cv2.imshow(self._window_name, self.overlay_image)
            # Manage key
            _last_key = cv2.waitKeyEx(1)
            if _last_key:
                self._last_key = _last_key

    def _generate_overlay_and_show_image(self):
        self._generate_overlay()
        if self._add_grid:
            self.add_grid()
        self._show_image()

    def set_image(self, image, bits_per_pixel=8, add_grid=False, grid_values=None, grid_colors=None, hide_overlay=False):
        self.input_image = image
        self._bits_per_pixel = bits_per_pixel
        self._hide_overlay = hide_overlay
        self._add_grid = add_grid
        if grid_values is not None:
            self._grid_values = grid_values
        if grid_colors is not None:
            self._grid_colors = grid_colors
        self._generate_overlay_and_show_image()
        _last_key = self._last_key
        self._last_key = 0
        return _last_key

    def save_image(self, saving_path, overlay=False):
        if overlay:
            image_to_save = self.overlay_image.copy()
        else:
            image_to_save = self.input_image.copy()
        # 10 bits is not supported for png format (only 8 or 16 bits)
        if self._bits_per_pixel == 10:
            image_to_save = (2 ** (16 - 10)) * image_to_save
        cv2.imwrite(saving_path, image_to_save)


if __name__ == '__main__':
    def generate_image(width=804, height=704, bit_width=np.uint8, iteration=0):
        image = np.zeros((height, width), bit_width)
        for i in range(width):
            for j in range(height):
                image[j, i] = (j+i+iteration) % 256
        return image

    smart_imshow = SmartImshow(os.path.basename(__file__))

    _i = 0

    while 1:
        # _i = (_i + 10) % 256
        test_image = generate_image(iteration=_i)
        key = smart_imshow.set_image(test_image)
        if key == SmartImshow.ESCAPE_KEY:
            break
