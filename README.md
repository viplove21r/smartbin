# ♻️ SmartBin – Intelligent Waste Classification System

SmartBin is an AI-powered waste classification application designed to identify different types of waste using Deep Learning and Computer Vision.

The application uses an **EfficientNetB0-based CNN model** to classify waste images into six categories. It also incorporates feedback-driven similarity matching to improve classification decisions using previously confirmed or corrected images.

The long-term goal is to integrate Fuzzy Logic to evaluate waste properties and assist in determining recyclability and biodegradability.

---

## 🚀 Features

### 1. AI-Based Waste Classification
- Upload an image of a waste item.
- Classifies waste into six categories:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- Displays the predicted category and model confidence.

### 2. Similarity-Based Classification
- Extracts 128-dimensional feature embeddings using the trained neural network.
- Compares uploaded images with previously reviewed images.
- Uses cosine similarity to identify visually similar waste samples.
- Uses weighted voting to determine a similarity-based classification.

### 3. Hybrid Classification
- Combines EfficientNet predictions with similarity-based classification.
- Produces a final classification using weighted scores.
- Allows comparison between the individual model prediction and similarity-based prediction.

### 4. User Feedback System
- Users can mark predictions as correct or incorrect.
- Incorrect predictions can be corrected by selecting the actual waste category.
- Stores feedback and feature embeddings for future similarity matching.

### 5. My Waste Dashboard
- Maintains waste scan history.
- Displays total scans, correct predictions, and incorrect predictions.
- Provides classification accuracy and feedback statistics.
- Displays individual scan details.

### 6. Compost Management
- Provides a compost management interface.
- Allows users to add and manage compostable items.
- Supports compost pit creation and management.

### 7. Django Admin
- Manage waste scan records.
- Inspect user feedback and actual classifications.
- Manage compost-related database records.

---

## 🧠 Machine Learning Model

SmartBin uses **Transfer Learning with EfficientNetB0**.

### Model Architecture

- Base model: EfficientNetB0
- Input size: 224 × 224 × 3
- Global Average Pooling
- Batch Normalization
- Dense layer: 256 neurons
- Dropout
- Dense layer: 128 neurons
- Dropout
- Output layer: 6 neurons with Softmax activation

### Training Approach

The model is trained in two stages:

1. **Feature Extraction:** The pretrained EfficientNetB0 backbone is frozen while the classification layers are trained.
2. **Fine-Tuning:** The final layers of the EfficientNetB0 backbone are unfrozen and fine-tuned with a lower learning rate.

Data augmentation and class weighting are used to improve generalization and handle class imbalance.

### Similarity Matching

The output of the 128-neuron Dense layer is used as a feature embedding.

Cosine similarity is calculated between the embedding of a newly uploaded image and previously stored feedback examples.

The most similar examples are selected, and weighted voting is used to obtain a similarity-based classification.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Django | Web framework |
| TensorFlow / Keras | Deep Learning |
| EfficientNetB0 | Image classification |
| OpenCV | Image processing |
| NumPy | Numerical computations |
| HTML | Frontend structure |
| CSS | Frontend styling |
| JavaScript | Frontend interactions |
| SQLite / PostgreSQL | Database |
| Git & GitHub | Version control |
| Render | Deployment |

---

## 📁 Project Structure

```text
garbage_project/
│
├── classifier/
│   ├── migrations/
│   ├── templates/
│   ├── models.py
│   ├── views.py
│   ├── admin.py
│   ├── urls.py
│   └── ...
│
├── garbage_project/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── models/
│   └── waste_classifier.keras
│
├── media/
├── static/
├── templates/
│
├── manage.py

```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd garbage_project
```

### 2. Create a Virtual Environment

```bash
python -m venv vipenv
```

Activate it:

**Ubuntu / Linux:**

```bash
source vipenv/bin/activate
```

**Windows:**

```bash
vipenv\Scripts\activate


### Configure Environment Variables

Configure the necessary environment variables, including your Django secret key and database settings.

For local development, configure SQLite in `settings.py` if PostgreSQL is not being used.

### Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

###  Create a Superuser

```bash
python manage.py createsuperuser
```

###  Run the Development Server

```bash
python manage.py runserver
```

Open the application:

```text
http://127.0.0.1:8000/
```

Django Admin:

```text
http://127.0.0.1:8000/admin/
```

---

## 📊 Dataset

The project uses a waste classification dataset containing six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The dataset is divided into training, validation, and testing sets.

| Dataset | Number of Images |
|---|---:|
| Training | 20,894 |
| Validation | 4,477 |
| Testing | 4,478 |
| **Total** | **29,849** |

---

## ☁️ Deployment

SmartBin can be deployed using Render.

General deployment steps:

1. Push the project to GitHub.
2. Create a PostgreSQL database on Render.
3. Create a Render Web Service connected to the GitHub repository.
4. Configure environment variables.
5. Set the build command to:

```bash
./build.sh
```

6. Set the start command to:

```bash
gunicorn garbage_project.wsgi:application
```

7. Deploy the application.

Refer to the [Render Django Deployment Guide](https://render.com/docs/deploy-django) for detailed instructions.

---

## 🔮 Future Improvements

- Integration of Fuzzy Logic for waste property evaluation.
- Recyclability and biodegradability assessment.
- OpenCV-based feature extraction and image analysis.
- Improved hybrid classification using optimized weights.
- Better handling of ambiguous and contaminated waste.
- Enhanced classification performance using additional datasets.
- Persistent cloud storage for uploaded images.



## 📜 License

This project is developed for educational and research purposes.
