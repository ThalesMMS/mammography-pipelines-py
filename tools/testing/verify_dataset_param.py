from mammography.io.dicom import create_mammography_image_from_dicom
import inspect

sig = inspect.signature(create_mammography_image_from_dicom)
result = 'dataset' in sig.parameters
print(result)
