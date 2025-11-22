#
# ui_viewer.py
# mammography-pipelines-py
#
# Matplotlib-based viewer that navigates DICOMs, previews ROIs, and saves annotated PNG crops.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Matplotlib-based viewer to browse DICOMs and save ROI crops as PNGs."""

import shutil
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import csv

from .data_manager import DataManager
from .dicom_loader import DicomImageLoader, apply_windowing
from .roi_selector import RoiSelector

class ImageViewerUI:
    """Interactive matplotlib UI to navigate DICOMs and save ROI crops."""
    
    def __init__(self, data_manager: DataManager):
        print("Initializing ImageViewerUI...")
        self.data_manager = data_manager
        self.dicom_loader = DicomImageLoader()
        self.roi_selector = RoiSelector(roi_size=448)

        self.current_folder_details = None
        self.current_dicom_index = 0
        self.dicom_pixel_data = None
        self.dicom_view_params = None

        self.fig = plt.figure(figsize=(13, 9))
        # Adjust subplot layout to make space for the preview sidebar
        self.fig.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.92)
        self.ax = self.fig.add_subplot(1, 1, 1)

        # Area for thumbnails of saved PNGs (top of the sidebar)
        self.png_sidebar_ax = self.fig.add_axes([0.77, 0.42, 0.21, 0.50])
        self.png_sidebar_ax.axis('off')

        # Area for live ROI preview (bottom of the sidebar)
        self.live_roi_preview_ax = self.fig.add_axes([0.77, 0.05, 0.21, 0.35])
        self.live_roi_preview_ax.axis('off')
        self._clear_live_roi_preview(message="Click the DICOM image to define an ROI.")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.roi_rect_patch = None
        self.save_status_text = None # Tracks the save confirmation message

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self.annotations_csv_path = os.path.join(project_root, "annotations.csv")
        self._initialize_annotations_csv()

        if self.data_manager.get_total_navigable_folders() > 0:
            print("Loading data from the first folder (after optional filtering)...")
            self.load_current_folder_data()
        else:
            print("No navigable folders found after filtering. The application cannot display images.")
            self.ax.text(0.5, 0.5, "No folders found\nfor the selected filters.",
                         ha='center', va='center', color='red', fontsize=12) 

    def _initialize_annotations_csv(self):
        header = ["AccessionNumber", "DCM_Filename", "Adjusted_ROI_Center_X", "Adjusted_ROI_Center_Y", "ROI_Size", "Saved_PNG_Filename"]
        if not os.path.exists(self.annotations_csv_path):
            try:
                with open(self.annotations_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                print(f"File '{self.annotations_csv_path}' created with header.")
            except IOError as e:
                print(f"Error creating or writing header to '{self.annotations_csv_path}': {e}")
                
    def _clear_live_roi_preview(self, message=None):
        self.live_roi_preview_ax.clear()
        if message:
            self.live_roi_preview_ax.text(0.5, 0.5, message,
                                          ha='center', va='center', fontsize=9, color='gray',
                                          transform=self.live_roi_preview_ax.transAxes)
        self.live_roi_preview_ax.axis('off')
        self.fig.canvas.draw_idle()
        
    def _update_live_roi_preview(self):
        if not self.roi_selector.is_defined() or self.dicom_pixel_data is None or self.dicom_view_params is None:
            self._clear_live_roi_preview("Preview unavailable")
            return

        roi_data_float = self.roi_selector.extract_roi_from_image(self.dicom_pixel_data)
        if roi_data_float is None:
            self._clear_live_roi_preview("Error generating preview")
            return

        roi_data_uint8 = apply_windowing(
            roi_data_float,
            self.dicom_view_params['wc'],
            self.dicom_view_params['ww'],
            self.dicom_view_params['photometric']
        ) 

        self.live_roi_preview_ax.clear()
        self.live_roi_preview_ax.imshow(roi_data_uint8, cmap='gray', aspect='auto')
        self.live_roi_preview_ax.set_title("Current ROI Preview", fontsize=9)
        self.live_roi_preview_ax.axis('off')
        self.fig.canvas.draw_idle()

    def load_current_folder_data(self):
        self.current_folder_details = self.data_manager.get_current_folder_details()
        if not self.current_folder_details:
            print("Error: Unable to obtain details for the current folder.")
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Error loading folder details.", ha='center', va='center', color='red')
            self.png_sidebar_ax.clear()
            self.png_sidebar_ax.axis('off')
            self.fig.canvas.draw_idle()
            self._clear_live_roi_preview("Error loading folder.")
            return

        if 'png_files' not in self.current_folder_details:
            self.current_folder_details['png_files'] = self.data_manager.get_png_files(
                self.current_folder_details['accession_number']
            )

        self.current_dicom_index = 0
        self.roi_selector.clear()
        self._remove_roi_rect_patch()
        self._clear_live_roi_preview("Click the image to define an ROI.")

        self.load_and_display_current_dicom()
        self._update_png_sidebar()

    def load_and_display_current_dicom(self):
        if not self.current_folder_details:
            self._clear_live_roi_preview()
            return

        dicom_files = self.current_folder_details.get("dicom_files", [])
        total_dicoms = len(dicom_files)
        
        if total_dicoms > 0:
            self.current_dicom_index = self.current_dicom_index % total_dicoms
        else:
            self.current_dicom_index = 0

        if not dicom_files:
            print("No DICOM files to display in this folder.")
            self.ax.clear()
            self.ax.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray') 
            self.update_title("No DICOM files to display")
            self.fig.canvas.draw_idle()
            self.dicom_pixel_data = None
            self.dicom_view_params = None
            self._clear_live_roi_preview("No DICOM files to display.")
            return

        dicom_filename = dicom_files[self.current_dicom_index]
        dicom_path = os.path.join(self.current_folder_details["folder_path"], dicom_filename)
        
        print(f"Loading DICOM: {dicom_filename}")
        pixel_data, view_params = self.dicom_loader.load_dicom_data(dicom_path)

        self.roi_selector.clear()
        self._remove_roi_rect_patch()
        self._clear_live_roi_preview("Click the image to define an ROI.")

        if pixel_data is not None and view_params is not None:
            self.dicom_pixel_data = pixel_data
            self.dicom_view_params = view_params
            self.display_image()
        else:
            print(f"Failed to load DICOM data: {dicom_filename}")
            self.ax.clear()
            self.ax.imshow(np.zeros((100,100), dtype=np.uint8), cmap='gray')
            self.update_title(f"Error loading {dicom_filename}")
            self.fig.canvas.draw_idle()
            self.dicom_pixel_data = None
            self.dicom_view_params = None
            self._clear_live_roi_preview("Error loading DICOM.")

    def display_image(self):
        if self.dicom_pixel_data is None or self.dicom_view_params is None:
            return

        self.ax.clear()

        wc = self.dicom_view_params['wc']
        ww = self.dicom_view_params['ww']
        photometric = self.dicom_view_params['photometric']

        vmin = wc - ww / 2.0
        vmax = wc + ww / 2.0
        cmap_to_use = 'gray_r' if photometric == 'MONOCHROME1' else 'gray'

        self.ax.imshow(self.dicom_pixel_data, cmap=cmap_to_use, vmin=vmin, vmax=vmax)
        self.ax.axis('on')

        dicom_files = self.current_folder_details.get("dicom_files", [])
        total_dicoms = len(dicom_files)
        dicom_index_display = 0
        if total_dicoms > 0:
            safe_index = self.current_dicom_index % total_dicoms
            dicom_index_display = safe_index + 1

        counter_text = f"Image: {dicom_index_display}/{total_dicoms}"
        self.ax.text(0.02, 0.02, counter_text,
                     color='yellow', fontsize=10,
                     transform=self.ax.transAxes,
                     bbox=dict(facecolor='black', alpha=0.5, pad=1))

        self.update_title()
        self._draw_roi_rect_patch()
        self.fig.canvas.draw_idle()

    def update_title(self, custom_message: str = None):
        """Update the window title with the current context."""
        if custom_message:
            self.fig.suptitle(custom_message)
            return

        if not self.current_folder_details:
            self.fig.suptitle("DICOM ROI Viewer")
            return

        folder_name = self.current_folder_details['accession_number']
        total_folders = self.data_manager.get_total_navigable_folders()
        current_index_display = self.data_manager.get_current_folder_index_display()

        dicom_files = self.current_folder_details.get("dicom_files", [])
        current_dicom_name = "N/A"
        total_dicoms = len(dicom_files)
        dicom_index_display = 0
        if total_dicoms > 0:
            safe_index = self.current_dicom_index % total_dicoms
            current_dicom_name = dicom_files[safe_index]
            dicom_index_display = safe_index + 1


        target = self.current_folder_details['target']
        laterality = self.current_folder_details['laterality']

        title = (f"Folder: {folder_name} ({current_index_display}/{total_folders}) | "
                 f"Img: {dicom_index_display}/{total_dicoms} [{current_dicom_name}] | "
                 f"Target: {target} | Lat: {laterality}")
        self.fig.suptitle(title, fontsize=10)

    # --- Event Handlers ---

    def on_key_press(self, event):
        """Keyboard event handler."""
        # print(f"Key pressed: {event.key}")

        if event.key == 'right':
            print("Navigating to next DICOM...")
            dicom_files = self.current_folder_details.get("dicom_files", [])
            if len(dicom_files) > 0:
                self.current_dicom_index += 1
                self.load_and_display_current_dicom()
            else:
                print("There are no additional DICOM files in this folder.")

        elif event.key == 'left':
            print("Navigating to previous DICOM...")
            dicom_files = self.current_folder_details.get("dicom_files", [])
            if len(dicom_files) > 0:
                self.current_dicom_index -= 1
                self.load_and_display_current_dicom()
            else:
                print("There are no additional DICOM files in this folder.")

        elif event.key == 'down':
            if self.data_manager.move_to_next_folder():
                print("Moving to the next folder...")
                self.load_current_folder_data()
            else:
                print("Already at the last folder.")
        elif event.key == 'up':
            if self.data_manager.move_to_previous_folder():
                print("Moving to the previous folder...")
                self.load_current_folder_data()
            else:
                print("Already at the first folder.")
        elif event.key == 'enter':
            self.save_roi()

    def on_mouse_click(self, event):
        if event.inaxes == self.ax and self.dicom_pixel_data is not None:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(f"Click detected at (x={x:.0f}, y={y:.0f})")
                img_height, img_width = self.dicom_pixel_data.shape[:2]
                self.roi_selector.set_center(int(x), int(y), img_height, img_width)
                self._draw_roi_rect_patch()
                self._update_live_roi_preview()
                self.fig.canvas.draw_idle()
            else:
                print("Click outside the image data area.")
        else:
            print("Click outside the main axis.")
            
    def _show_save_confirmation(self, message: str):
        self._clear_save_confirmation()
        self.save_status_text = self.fig.text(0.4, 0.01, message, ha='center', va='bottom',
                                              fontsize=10, color='green',
                                              bbox=dict(facecolor='lightgray', alpha=0.8, pad=3),
                                              transform=self.fig.transFigure)
        self.fig.canvas.draw_idle()

        # Timer to remove the message
        timer = self.fig.canvas.new_timer(interval=3000) # 3 seconds
        timer.add_callback(self._clear_save_confirmation)
        timer.start()
        self._save_confirmation_timer = timer # Keep a reference so it is not garbage-collected

    def _clear_save_confirmation(self):
        if hasattr(self, '_save_confirmation_timer'):
            self._save_confirmation_timer.stop()
            delattr(self, '_save_confirmation_timer')

        if self.save_status_text:
            try:
                self.save_status_text.remove()
            except Exception: # Be lenient if it was already removed
                pass
            self.save_status_text = None
            self.fig.canvas.draw_idle()
            
    # --- ROI Saving ---
    
    def save_roi(self):
        print("--- save_roi() called ---")
        self._clear_save_confirmation()

        if not self.roi_selector.is_defined():
            print("No ROI defined to save.")
            self._show_save_confirmation("Save failed: no ROI defined.")
            return

        if self.dicom_pixel_data is None or self.dicom_view_params is None:
            print("No image data loaded to extract the ROI.")
            self._show_save_confirmation("Save failed: no image loaded.")
            return

        if not self.current_folder_details:
            print("Error: Current folder details are unavailable.")
            self._show_save_confirmation("Save failed: missing folder details.")
            return


        roi_data_float = self.roi_selector.extract_roi_from_image(self.dicom_pixel_data)
        if roi_data_float is None:
            print("Failed to extract ROI data.")
            self._show_save_confirmation("Save failed: error extracting ROI.")
            return
            
        roi_data_uint8 = apply_windowing(
            roi_data_float,
            self.dicom_view_params['wc'],
            self.dicom_view_params['ww'],
            self.dicom_view_params['photometric']
        )

        dicom_files = self.current_folder_details.get("dicom_files", [])
        if not dicom_files:
            print("Error: Unable to determine the current DICOM file name (empty list).")
            self._show_save_confirmation("Save failed: no DICOM file.")
            return
        
        safe_index = self.current_dicom_index % len(dicom_files)
        current_dicom_filename = dicom_files[safe_index]
        base_dicom_filename_no_ext = os.path.splitext(current_dicom_filename)[0]
        
        # Generate an incremental PNG filename
        png_idx = 0
        while True:
            if png_idx == 0:
                png_filename_candidate = f"{base_dicom_filename_no_ext}.png"
            else:
                png_filename_candidate = f"{base_dicom_filename_no_ext}_{png_idx}.png"
            
            output_path_candidate = os.path.join(self.current_folder_details["folder_path"], png_filename_candidate)

            if not os.path.exists(output_path_candidate):
                png_filename = png_filename_candidate
                output_path = output_path_candidate
                break
            png_idx += 1
            if png_idx > 1000: # Safety limit
                print("Error: Too many PNG files with the same base name. Aborting.")
                self._show_save_confirmation("Save failed: file limit reached.")
                return

        print(f"Determined output path: {output_path}")

        try:
            pil_image = Image.fromarray(roi_data_uint8)
            pil_image.save(output_path)
            print(f"ROI saved successfully to: {output_path}")

            # Coordinates for the CSV (adjusted ROI center)
            adj_center_x, adj_center_y = None, None
            if self.roi_selector.current_roi_bounds:
                xmin, ymin, xmax, ymax = self.roi_selector.current_roi_bounds
                adj_center_x = (xmin + xmax) / 2.0
                adj_center_y = (ymin + ymax) / 2.0
            else: # Fallback, unlikely if is_defined() already passed
                adj_center_x = self.roi_selector.roi_center_x 
                adj_center_y = self.roi_selector.roi_center_y
                print("Warning: current_roi_bounds unavailable for adjusted center, using original click.")


            # Append entry to the CSV
            try:
                accession_number = self.current_folder_details['accession_number']
                roi_size = self.roi_selector.roi_size
                
                csv_row = [accession_number, current_dicom_filename, adj_center_x, adj_center_y, roi_size, png_filename]
                with open(self.annotations_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)
                print(f"Annotation registered in '{self.annotations_csv_path}'")
            except KeyError as ke:
                print(f"Key error while preparing data for CSV: {ke}. Check 'current_folder_details' and 'roi_selector'.")
            except IOError as e:
                print(f"I/O error while writing to '{self.annotations_csv_path}': {e}")
            except Exception as e_csv:
                print(f"Unexpected error while recording ROI data to the CSV: {e_csv}")


            if png_filename not in self.current_folder_details['png_files']:
                self.current_folder_details['png_files'].append(png_filename)
                self.current_folder_details['png_files'].sort()
            self._update_png_sidebar()

            self.fig.canvas.draw_idle()
            self._show_save_confirmation(f"Saved: {png_filename}")


        except Exception as e:
            print(f"Error saving PNG at '{output_path}': {e}")
            self._show_save_confirmation(f"Error saving PNG: {e}")

    def _draw_roi_rect_patch(self):
        self._remove_roi_rect_patch()
        if self.roi_selector.is_defined():
            rect_coords = self.roi_selector.get_current_roi_display_rect()
            if rect_coords:
                x, y, w, h = rect_coords
                self.roi_rect_patch = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1, edgecolor='lime', facecolor='none', label='ROI'
                )
                self.ax.add_patch(self.roi_rect_patch)

    def _remove_roi_rect_patch(self):
        if self.roi_rect_patch:
            try:
                self.roi_rect_patch.remove()
            except ValueError:
                pass
            self.roi_rect_patch = None

    def _update_png_sidebar(self):
        self.png_sidebar_ax.clear()
        self.png_sidebar_ax.axis('off')
        self.png_sidebar_ax.set_title("Saved ROIs (.png)", fontsize=9)
        self.png_sidebar_ax.set_ylim(0, 1)

        if not self.current_folder_details:
            return

        png_files = self.current_folder_details.get('png_files', [])
        
        if not png_files:
            self.png_sidebar_ax.text(0.5, 0.5, "No PNG\nannotations\nsaved.", 
                                     ha='center', va='center', fontsize=8, color='gray',
                                     transform=self.png_sidebar_ax.transAxes)
            self.fig.canvas.draw_idle() 
            return

        num_pngs = len(png_files)
        max_thumbnails_to_display = 6 # Tuned for the sidebar height
        png_files_to_display = png_files[:max_thumbnails_to_display]
        num_display = len(png_files_to_display)

        total_vertical_space = 0.95 
        spacing = 0.02 
        thumbnail_slot_height = 0

        if num_display > 0:
            thumbnail_slot_height = (total_vertical_space - max(0, num_display - 1) * spacing) / num_display
            if thumbnail_slot_height <= 0: thumbnail_slot_height = 0.05
        
        if num_pngs > max_thumbnails_to_display:
            self.png_sidebar_ax.text(0.5, 0.01, f"(showing {max_thumbnails_to_display}/{num_pngs})", 
                                     ha='center', va='bottom', fontsize=7, color='gray',
                                     transform=self.png_sidebar_ax.transAxes)

        current_y_top = 0.98 

        for i, png_filename in enumerate(png_files_to_display):
            file_path = os.path.join(self.current_folder_details['folder_path'], png_filename)
            try:
                img = Image.open(file_path).convert('L') 
                img_array = np.array(img)

                y_bottom = current_y_top - thumbnail_slot_height 
                
                img_aspect_ratio = img.width / img.height
                thumb_width_normalized = thumbnail_slot_height * img_aspect_ratio                
                thumb_xmin = (1.0 - thumb_width_normalized) / 2.0
                thumb_xmax = thumb_xmin + thumb_width_normalized
                thumb_xmin = max(0, thumb_xmin)
                thumb_xmax = min(1, thumb_xmax)
                
                actual_height_used = thumbnail_slot_height 
                
                if thumb_width_normalized > 1.0:
                    thumb_width_normalized = 1.0
                    thumb_xmin = 0.0
                    thumb_xmax = 1.0
                    new_thumb_height = thumb_width_normalized / img_aspect_ratio
                    y_bottom = current_y_top - new_thumb_height 
                    actual_height_used = new_thumb_height 
                
                self.png_sidebar_ax.imshow(img_array, 
                                           cmap='gray',        
                                           aspect='equal',     
                                           extent=(thumb_xmin, thumb_xmax, y_bottom, current_y_top))
                
                text_y_position = y_bottom - 0.015 
                self.png_sidebar_ax.text((thumb_xmin + thumb_xmax) / 2, text_y_position, 
                                         os.path.basename(png_filename), 
                                         ha='center', va='top', fontsize=6, color='black') 
                
                current_y_top = y_bottom - spacing 

            except FileNotFoundError:
                error_msg = f"Error:\n{png_filename}\nnot found"
                self.png_sidebar_ax.text(0.5, current_y_top - thumbnail_slot_height / 2.0, 
                                         error_msg, ha='center', va='center', fontsize=6, color='red')
                current_y_top = current_y_top - thumbnail_slot_height - spacing 
            except Exception as e:
                error_msg = f"Error loading\n{png_filename}"
                self.png_sidebar_ax.text(0.5, current_y_top - thumbnail_slot_height / 2.0, 
                                         error_msg, ha='center', va='center', fontsize=6, color='red')
                current_y_top = current_y_top - thumbnail_slot_height - spacing 
        
        self.fig.canvas.draw_idle()
        
    def show(self):
        plt.show()
