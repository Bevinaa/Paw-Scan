# 🐾 Paw Scan

**Paw Scan** is a Flutter-based mobile application that classifies cat and dog breeds using deep learning transformer models. It acts as the mobile frontend for the project [Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture](https://github.com/Bevinaa/Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture), where image classification is handled via a backend hosted on Render.

> ⚠️ This version requires an internet connection to send images to a web API. Hosted on Render's free tier — may trigger memory limits if usage is high.

---

## ✨ Features

-  Capture or select an image of a cat or dog.
-  Backend uses ViT, Swin, and DeiT transformer models for breed classification.
-  Displays predicted breed name and confidence.
-  Communicates with a hosted Python API for inference.

---

## 🛠 Tech Stack

### Frontend
- **Flutter** (Dart)
- **Material Design UI**
- **HTTP** for backend communication

### Backend (Linked Project)
- **PyTorch** with `timm` transformer models
- **Residual MLP** for classification
- **Flask/FastAPI** for API endpoints
- **Hosted on**: [Render](https://render.com)

---

##  Getting Started
> ⚠️ Ensure that you have your mobile phone connected as a emulator using 'flutter devices' in cmd.
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bevinaa/Paw-Scan.git
   cd Paw-Scan
   ```
2. **Install Flutter packages**:
   ```bash
   flutter pub get
   ```
3. **Set your backend API URL from Render**:
   ```bash
   'https://your-api-url.onrender.com/predict';
   ```
4. **Run the app**:
   ```bash
   flutter run
   flutter build apk
  
---

## Related Repository

[Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture](https://github.com/Bevinaa/Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture)

---

## Mobile Application - Overview 

| UI of the App | Prediction |
|------------|--------------|
| ![WhatsApp Image 2025-07-07 at 12 35 54_ec04ad7d](https://github.com/user-attachments/assets/20b61622-68b9-41bc-a921-897367f6cb82) | ![image](https://github.com/user-attachments/assets/4f2b1f02-1fe6-4146-b9d9-e815a7c738a4) |

---

## Contributions

Contributions, feedback, and issues are welcome! Feel free to fork the repository and submit a pull request.

---

## Contact

For questions or further information, please contact:

- **Name:** Bevina R
- **Email:** bevina2110@gmail.com
- **GitHub:** [Bevinaa](https://github.com/Bevinaa)

---

