import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = "my_digits/raw"
OUT_DIR = "my_digits/processed"


def binarize(gray, block_size, c):
    # Blur removes paper texture; adaptive threshold handles uneven lighting/shadows
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV -> digits become white on black, like MNIST
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, c)


def find_digit_boxes(binary, min_area):
    h, w = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):  # label 0 is background
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if area < min_area or touches_border:  # drop noise specks and paper edges
            continue
        boxes.append([x, y, x + bw, y + bh])

    # Merge components that overlap horizontally (a digit written in two strokes, e.g. 5 or 4)
    boxes.sort(key=lambda b: b[0])
    merged = []
    for b in boxes:
        if merged and b[0] <= merged[-1][2]:
            m = merged[-1]
            merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
        else:
            merged.append(b)

    # Drop leftover tiny blobs relative to typical digit height
    if merged:
        med_h = np.median([b[3] - b[1] for b in merged])
        merged = [b for b in merged if (b[3] - b[1]) >= 0.25 * med_h]
    return merged


def to_mnist(binary_crop, thickness_ratio):
    # 1) Thicken strokes: pen lines are much thinner than MNIST strokes relative to digit size
    k = max(1, int(round(max(binary_crop.shape) * thickness_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    padded = cv2.copyMakeBorder(binary_crop, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    thick = cv2.dilate(padded, kernel)
    ys, xs = np.nonzero(thick)
    thick = thick[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2) Fit into a 20x20 box keeping aspect ratio (MNIST convention)
    h, w = thick.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(thick, (nw, nh), interpolation=cv2.INTER_AREA)

    # 3) Paste into 28x28 canvas, then shift so the center of mass is at the center
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    m = cv2.moments(canvas)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    return cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)


def save_debug(img, binary, boxes, digits, out_path):
    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    n = max(len(digits), 1)
    fig = plt.figure(figsize=(max(12, n * 1.3), 9))
    gs = fig.add_gridspec(3, n, height_ratios=[3, 3, 1.4])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title("detected digits")
    ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(binary, cmap="gray")
    ax.set_title("binary (adaptive threshold)")
    ax.axis("off")
    for i, d in enumerate(digits):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(d, cmap="gray")
        ax.set_title(str(i))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.path.join(RAW_DIR, "number.jpg"))
    p.add_argument("--block-size", type=int, default=51)    # adaptive threshold window, must be odd
    p.add_argument("--c", type=int, default=15)             # higher -> less noise, thinner strokes
    p.add_argument("--min-area", type=int, default=40)      # components smaller than this are noise
    p.add_argument("--thickness", type=float, default=0.08) # dilation kernel size relative to digit size
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray, args.block_size, args.c)
    boxes = find_digit_boxes(binary, args.min_area)

    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):  # clear outputs from previous runs
        if f.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))

    digits = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        d = to_mnist(binary[y1:y2, x1:x2], args.thickness)
        cv2.imwrite(os.path.join(OUT_DIR, f"digit_{i:02d}.png"), d)
        digits.append(d)

    save_debug(img, binary, boxes, digits, os.path.join("my_digits", "preprocess_debug.png"))
    print(f"found {len(digits)} digits -> {OUT_DIR}/")
    print("debug image -> my_digits/preprocess_debug.png")


if __name__ == "__main__":
    main()