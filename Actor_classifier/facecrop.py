from pathlib import Path
import cv2

src = Path("data")
dst = Path("data_faces")

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

for actor_dir in src.iterdir():

    if not actor_dir.is_dir():
        continue

    out_dir = dst / actor_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in actor_dir.glob("*"):

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) == 0:
            continue

        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        crop = img[y:y+h, x:x+w]

        cv2.imwrite(
            str(out_dir / img_path.name),
            crop
        )