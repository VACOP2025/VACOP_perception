import cv2
import numpy as np
import glob
import os


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_calib_cam/dataset_cam")  # path to the calibration board images
PATTERN = "photo_*.jpg" 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "calib_parameters/cam_intrinsics")


# Number of INNER CORNERS
# Example: 9x6 square checkerboard -> (8,5) inner corners
CHECKERBOARD = (7, 5) 

# Size of one square (here 10 cm = 0.10 m or 100 mm)
SQUARE_SIZE = 0.10  # in meters
# =========================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# 3D points in the checkerboard world frame (z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = sorted(glob.glob(os.path.join(DATASET_DIR, PATTERN)))
if not images:
    raise SystemExit(f"No images found in {DATASET_DIR} matching {PATTERN}")

print(f"Images found: {len(images)}")

img_size = None
used = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Cannot read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])  # (w,h)

    # Corner detection
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        used += 1
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
    else:
        print(f"Checkerboard not detected: {os.path.basename(fname)}")

print(f"Images used: {used} / {len(images)}")
if used < 8:
    raise SystemExit("Not enough usable images. Aim for at least 10-15 images with the board clearly visible.")

# Calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# Mean reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
mean_error = total_error / len(objpoints)

print("\n===== RESULTATS =====")
print("RMS (cv2.calibrateCamera ret):", ret)
print("Intrinsic matrix K:\n", K)
print("Distortion [k1 k2 p1 p2 k3 ...]:\n", dist.ravel())
print("Mean reprojection error (px):", mean_error)

# Save (OpenCV YAML + NPZ)
fs = cv2.FileStorage(os.path.join(OUTPUT_DIR, "calibration.yml"), cv2.FILE_STORAGE_WRITE)
fs.write("K", K)
fs.write("dist", dist)
fs.write("image_width", img_size[0])
fs.write("image_height", img_size[1])
fs.release()

np.savez("calibration.npz", K=K, dist=dist, width=img_size[0], height=img_size[1])

print("\nSaved: calibration.yml and calibration.npz")