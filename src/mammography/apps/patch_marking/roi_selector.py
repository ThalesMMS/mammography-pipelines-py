#
# roi_selector.py
# mammography-pipelines
#
# Utility class that maintains square ROIs centered on clicks while clamping to image bounds.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Utility class to compute square ROIs and keep them within image boundaries."""

import numpy as np

class RoiSelector:
    """Maintain a square ROI centered on user clicks while clamping to image bounds."""
    def __init__(self, roi_size: int = 448):  # 448 is the default size
        """
        Initialize the RoiSelector.

        Args:
            roi_size (int): Side length of the square ROI (default: 448).
        """
        if roi_size <= 0:
            raise ValueError("ROI size must be a positive integer.")
        self.roi_size = roi_size
        self.roi_center_x = None
        self.roi_center_y = None
        self.image_height = None
        self.image_width = None
        
        self.current_roi_bounds = None  # Stores (xmin, ymin, xmax, ymax) for the current ROI

    def set_center(self, center_x: int, center_y: int, image_height: int, image_width: int):
        """
        Define the ROI center and the dimensions of the image it applies to,
        then recompute ROI bounds.

        Args:
            center_x (int): X coordinate of the ROI center (column).
            center_y (int): Y coordinate of the ROI center (row).
            image_height (int): Height of the source image.
            image_width (int): Width of the source image.
        """
        self.roi_center_x = int(round(center_x))
        self.roi_center_y = int(round(center_y))
        self.image_height = image_height
        self.image_width = image_width
        self._calculate_bounds()

    def _calculate_bounds(self):
        """
        Calculate ROI bounds (xmin, ymin, xmax, ymax) based on the center and size,
        ensuring the ROI stays inside image borders. Bounds are for slicing
        (xmax and ymax are exclusive).
        """
        if self.roi_center_x is None or self.roi_center_y is None or \
           self.image_height is None or self.image_width is None:
            self.current_roi_bounds = None
            return

        half_size = self.roi_size // 2
        
        # Compute initial bounds
        xmin = self.roi_center_x - half_size
        xmax = self.roi_center_x + (self.roi_size - half_size)  # Ensures the span equals roi_size
        ymin = self.roi_center_y - half_size
        ymax = self.roi_center_y + (self.roi_size - half_size)

        # Adjust bounds to stay within image limits (clipping)
        # For slicing, xmax and ymax are exclusive.
        # To keep a fixed ROI size when possible, if one side is clipped we extend the other
        # side when space allows, otherwise the ROI is shifted. For simplicity we prioritize
        # the center and clip at the borders. The resulting ROI may be smaller than self.roi_size
        # if the center is close to the edge.

        # For a fixed-size ROI that shifts:
        if xmin < 0:
            xmax -= xmin # xmax = xmax + abs(xmin)
            xmin = 0
        if ymin < 0:
            ymax -= ymin # ymax = ymax + abs(ymin)
            ymin = 0

        if xmax > self.image_width:
            xmin -= (xmax - self.image_width)
            xmax = self.image_width
        if ymax > self.image_height:
            ymin -= (ymax - self.image_height)
            ymax = self.image_height
            
        # Ensure xmin/ymin are not negative after adjustments
        xmin = max(0, xmin)
        ymin = max(0, ymin)

        # Ensure xmax/ymax are not smaller than xmin/ymin
        xmax = max(xmin, xmax)
        ymax = max(ymin, ymax)

        self.current_roi_bounds = (int(xmin), int(ymin), int(xmax), int(ymax))

    def get_current_roi_display_rect(self) -> tuple[int, int, int, int] | None:
        """
        Return coordinates (x, y, width, height) to draw a rectangle for the ROI.
        Useful for GUI libraries such as Matplotlib (patches.Rectangle).

        Returns:
            tuple[int, int, int, int] | None: (xmin, ymin, width, height) or None if undefined.
        """
        if self.current_roi_bounds:
            xmin, ymin, xmax, ymax = self.current_roi_bounds
            width = xmax - xmin
            height = ymax - ymin
            return (xmin, ymin, width, height)
        return None

    def extract_roi_from_image(self, full_image_array: np.ndarray) -> np.ndarray | None:
        """
        Extract the ROI from the provided image using the calculated bounds.
        The extracted image has size self.roi_size x self.roi_size, filled with zeros
        if the original ROI sits near an edge and results in a smaller slice.

        Args:
            full_image_array (np.ndarray): Full image array.

        Returns:
            np.ndarray | None: ROI array (self.roi_size x self.roi_size), or None if undefined.
        """
        if self.current_roi_bounds is None or full_image_array is None:
            return None

        xmin, ymin, xmax, ymax = self.current_roi_bounds
        
        # Extract portion of the original image
        extracted_slice = full_image_array[ymin:ymax, xmin:xmax]

        # Create a canvas of the final ROI size and fill with zeros.
        # Data type matches the original image.
        final_roi_array = np.zeros((self.roi_size, self.roi_size), dtype=full_image_array.dtype)

        # Determine placement of the extracted slice inside the final canvas.
        # If the slice is smaller due to border clipping, it will stay in the
        # upper-left corner; the remainder stays zero-filled.
        slice_height, slice_width = extracted_slice.shape[:2]

        # Ensure we never write beyond final_roi_array, even if slice height/width
        # are unexpectedly larger than self.roi_size (should not happen with current logic).
        copy_height = min(slice_height, self.roi_size)
        copy_width = min(slice_width, self.roi_size)
        
        final_roi_array[0:copy_height, 0:copy_width] = extracted_slice[0:copy_height, 0:copy_width]
        
        return final_roi_array

    def is_defined(self) -> bool:
        """Return True if an ROI is defined (center has been set)."""
        return self.current_roi_bounds is not None

    def clear(self):
        """Clear the current ROI definition."""
        self.roi_center_x = None
        self.roi_center_y = None
        # Keep dimensions unset to force recalculation on next set_center call.
        self.image_height = None
        self.image_width = None
        self.current_roi_bounds = None

# Example usage (for quick manual testing only):
if __name__ == '__main__':
    # Simulate an image
    img_height, img_width = 400, 500
    # Use the float array that DicomImageLoader would return
    # (for this test, a zero-filled array with the correct dtype)
    mock_image_data = np.zeros((img_height, img_width), dtype=np.float32) 

    selector = RoiSelector(roi_size=448)  # ROI size

    print("ROI Selector created.")
    print(f"ROI defined? {selector.is_defined()}")

    # Test 1: Center of the image
    center_x_test1, center_y_test1 = 250, 200
    selector.set_center(center_x_test1, center_y_test1, img_height, img_width)
    print(f"\nROI defined after set_center? {selector.is_defined()}")
    bounds_rect = selector.get_current_roi_display_rect()
    if bounds_rect:
        print(f"Test 1 (Center): Display bounds (x,y,w,h): {bounds_rect}")
        extracted_roi = selector.extract_roi_from_image(mock_image_data)
        if extracted_roi is not None:
            print(f"Test 1: Extracted ROI shape: {extracted_roi.shape}")
    selector.clear()

    # Test 2: Near the top-left corner (expect clipping and padding)
    center_x_test2, center_y_test2 = 30, 40  # ROI 448x448 centered at (30, 40)
                                            # xmin = 30-64 = -34 -> 0
                                            # ymin = 40-64 = -24 -> 0
                                            # xmax = 30+64 = 94
                                            # ymax = 40+64 = 104
    selector.set_center(center_x_test2, center_y_test2, img_height, img_width)
    bounds_rect = selector.get_current_roi_display_rect()
    if bounds_rect:
        print(f"\nTest 2 (Top-left corner): Display bounds (x,y,w,h): {bounds_rect}")
        # Expected final bounds: (0, 0, 448, 448)
        extracted_roi = selector.extract_roi_from_image(mock_image_data)
        if extracted_roi is not None:
            print(f"Test 2: Extracted ROI shape: {extracted_roi.shape}")
            # Ensure the ROI remains zero-filled (mock_image_data is zeros)
            assert np.all(extracted_roi == 0) 
    selector.clear()

    # Test 3: Near the bottom-right corner
    center_x_test3, center_y_test3 = img_width - 20, img_height - 30  # x=480, y=370
                                                                      # Expected bounds: (372, 272, 448, 448)
    selector.set_center(center_x_test3, center_y_test3, img_height, img_width)
    bounds_rect = selector.get_current_roi_display_rect()
    if bounds_rect:
        print(f"\nTest 3 (Bottom-right corner): Display bounds (x,y,w,h): {bounds_rect}")
        extracted_roi = selector.extract_roi_from_image(mock_image_data)
        if extracted_roi is not None:
            print(f"Test 3: Extracted ROI shape: {extracted_roi.shape}")
    selector.clear()

    print(f"\nROI defined after clear? {selector.is_defined()}")
