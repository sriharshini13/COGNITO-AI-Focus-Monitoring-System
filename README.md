\# COGNITO — AI Focus Monitoring System



> \*\*An AI-powered real-time focus and productivity monitoring system using Computer Vision and Machine Learning.\*\*



COGNITO is a real-time AI-based focus monitoring system designed to analyze a user's visual behavioral cues through a webcam and convert them into actionable focus, attention, drowsiness, emotion, distraction, and productivity insights.



The system combines \*\*MediaPipe facial landmark detection, OpenCV, Random Forest classification, behavioral feature engineering, real-time analytics, and a Flask-based web application\*\* to create an interactive monitoring environment for students, remote workers, and other focused-work scenarios.



\---



\## 🚀 What COGNITO Does



COGNITO processes webcam frames and extracts behavioral features from facial landmarks, including:



\* Eye Aspect Ratio (\*\*EAR\*\*)

\* Mouth Aspect Ratio (\*\*MAR\*\*)

\* Gaze position

\* Head pitch

\* Head yaw



These features are passed through trained machine-learning models to estimate:



\* \*\*Focus level:\*\* High / Medium / Low

\* \*\*Emotion state:\*\* Engaged / Bored / Confused / Frustrated

\* \*\*Attention score\*\*

\* \*\*Drowsiness severity\*\*

\* \*\*Productivity score\*\*

\* \*\*Distraction count\*\*

\* \*\*Focus streak\*\*

\* \*\*Session-level performance\*\*



The system then aggregates these signals into a persistent session record and presents the results through an interactive web dashboard.



\---



\## ✨ Key Features



\### 🧠 AI-Based Focus Classification



COGNITO uses a \*\*Random Forest classifier\*\* to classify focus into:



```text

High

Medium

Low

```



The focus model is trained using six engineered behavioral features:



```text

EAR

MAR

Gaze\_X

Gaze\_Y

Head\_Pitch

Head\_Yaw

```



The training pipeline uses:



\* Stratified train/test splitting

\* 80/20 train-test split

\* SMOTE-based training-set balancing

\* Random Forest classification

\* Classification report

\* Confusion matrix

\* Feature importance analysis



The trained model is stored as:



```text

models/cognito\_model.pkl

```



\---



\### 😊 Emotion / Engagement Classification



COGNITO includes a second Random Forest model for behavioral/emotional-state classification.



The training pipeline maps the available behavioral labels into four application-level states:



```text

Engaged

Bored

Confused

Frustrated

```



The same six facial-behavior features are used by the model.



The trained model is stored as:



```text

models/emotion\_model.pkl

```



\---



\### 👁️ Facial Landmark-Based Feature Engineering



MediaPipe Face Landmarker is used to obtain facial landmarks from the webcam stream.



COGNITO derives:



| Feature    | Purpose                                |

| ---------- | -------------------------------------- |

| EAR        | Eye openness / drowsiness estimation   |

| MAR        | Mouth opening / yawning-related signal |

| Gaze X/Y   | Approximate gaze direction             |

| Head Pitch | Vertical head orientation              |

| Head Yaw   | Horizontal head orientation            |



This feature vector becomes the input to the machine-learning models.



```text

Webcam Frame

&#x20;     ↓

MediaPipe Face Landmarker

&#x20;     ↓

Facial Landmarks

&#x20;     ↓

Feature Extraction

&#x20;     ↓

EAR / MAR / Gaze / Head Pose

&#x20;     ↓

ML Models

```



\---



\### 📊 Real-Time Productivity Score



COGNITO converts multiple behavioral signals into a \*\*0–100 productivity score\*\*.



The current implementation combines:



```text

Focus       → 40%

Emotion     → 25%

Attention   → 25%

Posture     → 10%

```



The resulting score is continuously tracked throughout a session.



The system also calculates:



\* Current productivity score

\* Average productivity

\* Peak productivity

\* Score timeline

\* Productivity grade



Grades are currently mapped as:



```text

A → Excellent

B → Good Job

C → Average

D → Needs Improvement

F → Poor Focus

```



\---



\### 😴 Drowsiness Detection



COGNITO uses \*\*Eye Aspect Ratio (EAR)\*\* as a drowsiness signal.



The system categorizes eye-state severity into levels such as:



```text

Awake

Mild Drowsiness

Moderate Drowsiness

Severe Drowsiness

```



Thresholds can also vary according to the selected user profile.



\---



\### 🥱 Yawning Detection



MAR-based mouth-opening analysis is used to detect possible yawning events.



When yawning/fatigue-related behavior is detected, COGNITO can provide an appropriate visual recommendation such as taking an eye/rest break.



\---



\### 🎯 Attention Monitoring



Attention is derived from the same visual behavioral signals, including:



\* Eye openness

\* Mouth state

\* Gaze position

\* Head orientation



The resulting attention value is converted into an interpretable attention level for the monitoring dashboard.



\---



\### ⚠️ Distraction Tracking



COGNITO tracks transitions into the \*\*Low Focus\*\* state.



A cooldown mechanism prevents a continuous period of low focus from being counted as multiple separate distractions.



The system records:



\* Total distractions

\* Distraction timestamps

\* Distractions per hour



This allows session-level distraction behavior to be analyzed instead of simply displaying instantaneous predictions.



\---



\### 🔔 Smart Alerts \& Break Suggestions



COGNITO contains dedicated modules for:



\* Smart alerts

\* Break suggestions

\* Drowsiness warnings

\* Low-focus warnings

\* Fatigue-related notifications



Different monitoring profiles use different alert thresholds and break intervals.



\---



\## 👤 Personalized Monitoring Profiles



The web application supports multiple monitoring profiles.



\### 🎓 Student



Designed for:



\* Studying

\* E-learning

\* Online classes



Tracks:



```text

Focus

Emotion

Attention

Productivity

Distractions

```



\### 💻 Remote Worker



Designed for:



\* Work-from-home sessions

\* Productivity monitoring



\### 🚗 Driver



Focused on:



\* Alertness

\* Drowsiness

\* Gaze-related monitoring



\### 🎮 Gamer



Focused on:



\* Focus

\* Attention

\* Distraction monitoring



\### 🏥 Medical Pro



Configured for:



\* Alertness monitoring

\* Drowsiness-related signals



\### 🏃 Athlete



Focused on:



\* Mental focus

\* Attention

\* Emotion



Each profile has its own behavioral thresholds, alert messages, metrics, and break configuration.



\---



\# 🌐 Web Application



COGNITO includes a Flask-based web application that turns the underlying computer-vision pipeline into a complete user-facing system.



The application includes:



\### 🔐 Authentication



Users can:



\* Register

\* Log in

\* Log out

\* Maintain a personal profile



Passwords are stored using Werkzeug password hashing.



\---



\### 🧭 Onboarding



New users are guided through onboarding and can select their default monitoring profile.



\---



\### 📹 Live Monitoring



The monitoring interface provides real-time access to:



\* Focus classification

\* Focus confidence

\* Emotion classification

\* Emotion confidence

\* Attention score

\* Drowsiness level

\* Productivity score

\* Distraction count

\* Focus streak

\* Session duration

\* Alerts

\* Break suggestions



\---



\### 📈 Dashboard



The dashboard provides session-level analytics and personal performance information.



COGNITO maintains:



\* Session history

\* Productivity trends

\* Focus distribution

\* Emotion distribution

\* Distraction statistics

\* Personal bests

\* Focus streaks



\---



\### 📋 Session Reports



Each completed monitoring session can store information such as:



\* Date

\* Time

\* Duration

\* Average productivity

\* Productivity grade

\* Focus distribution

\* Emotion distribution

\* Total distractions

\* Distractions per hour

\* Yawn count

\* Maximum focus streak



\---



\### 📤 CSV Export



Session information can be exported as CSV data for further analysis.



The exported information includes metrics such as:



```text

Date

Time

Duration

Average Productivity

Grade

High Focus %

Medium Focus %

Low Focus %

Engaged %

Bored %

Total Distractions

Distractions / Hour

Yawn Count

```



\---



\### 🏆 Gamification



COGNITO includes a lightweight gamification layer.



Users can earn points through:



\* Completing sessions

\* Answering focus quizzes

\* Maintaining quiz streaks



A leaderboard ranks users according to accumulated points.



\---



\### 🧠 Focus Check Quizzes



COGNITO includes profile-specific quizzes designed to introduce active engagement during monitoring sessions.



Quiz functionality includes:



\* Multiple-choice questions

\* Correct/incorrect feedback

\* Points

\* Quiz streaks

\* Explanations



This creates an additional interaction layer instead of making the application purely passive monitoring.



\---



\### 👨‍💼 Admin Dashboard



The application includes an administrative view for monitoring users and active sessions.



Administrators can view information including:



\* Registered users

\* Active sessions

\* Current focus state

\* Attention score

\* Productivity score

\* Drowsiness level

\* Selected profile

\* Session activity



The admin interface also supports sending alerts to active users.



\---



\# 🏗️ System Architecture



```mermaid

flowchart TD



&#x20;   A\[Webcam] --> B\[OpenCV Frame Capture]



&#x20;   B --> C\[MediaPipe Face Landmarker]



&#x20;   C --> D\[Facial Landmark Extraction]



&#x20;   D --> E\[Feature Engineering]



&#x20;   E --> E1\[EAR]

&#x20;   E --> E2\[MAR]

&#x20;   E --> E3\[Gaze X/Y]

&#x20;   E --> E4\[Head Pitch/Yaw]



&#x20;   E1 --> F\[Focus Random Forest]

&#x20;   E2 --> F

&#x20;   E3 --> F

&#x20;   E4 --> F



&#x20;   E1 --> G\[Emotion Random Forest]

&#x20;   E2 --> G

&#x20;   E3 --> G

&#x20;   E4 --> G



&#x20;   E --> H\[Attention Analysis]

&#x20;   E1 --> I\[Drowsiness Analysis]

&#x20;   E2 --> J\[Yawning Detection]



&#x20;   F --> K\[Focus State]

&#x20;   G --> L\[Emotion State]

&#x20;   H --> M\[Attention Score]

&#x20;   I --> N\[Drowsiness Level]

&#x20;   J --> O\[Fatigue Signal]



&#x20;   K --> P\[Productivity Engine]

&#x20;   L --> P

&#x20;   M --> P



&#x20;   P --> Q\[Productivity Score]



&#x20;   K --> R\[Distraction Tracker]

&#x20;   K --> S\[Focus Streak]

&#x20;   N --> T\[Smart Alerts]

&#x20;   O --> T



&#x20;   Q --> U\[Session Analytics]

&#x20;   R --> U

&#x20;   S --> U

&#x20;   T --> U



&#x20;   U --> V\[Flask Web Application]



&#x20;   V --> V1\[Live Monitor]

&#x20;   V --> V2\[Dashboard]

&#x20;   V --> V3\[Reports]

&#x20;   V --> V4\[Leaderboard]

&#x20;   V --> V5\[Admin Panel]

```



\---



\# 🔬 Machine Learning Pipeline



\## 1. Feature Extraction



The feature extraction pipeline generates a structured representation of facial behavior from video frames.



The current feature vector is:



```text

\[

&#x20;   EAR,

&#x20;   MAR,

&#x20;   Gaze\_X,

&#x20;   Gaze\_Y,

&#x20;   Head\_Pitch,

&#x20;   Head\_Yaw

]

```



\---



\## 2. Focus Model



The focus model follows:



```text

Feature Dataset

&#x20;     ↓

Train/Test Split

&#x20;     ↓

SMOTE on Training Data

&#x20;     ↓

Random Forest

&#x20;     ↓

Evaluation

&#x20;     ↓

Saved Model

```



The implementation uses:



```text

RandomForestClassifier

n\_estimators = 100

max\_depth = 10

random\_state = 42

```



The trained model is saved using Joblib.



\---



\## 3. Emotion Model



The emotion model follows a similar pipeline:



```text

Behavioral Features

&#x20;     ↓

DAiSEE-derived Labels

&#x20;     ↓

Emotion Mapping

&#x20;     ↓

Train/Test Split

&#x20;     ↓

SMOTE

&#x20;     ↓

Random Forest

&#x20;     ↓

Evaluation

&#x20;     ↓

Saved Model

```



The four application classes are:



```text

Engaged

Bored

Confused

Frustrated

```



\---



\# 🧮 Productivity Scoring Logic



The current scoring engine combines four components:



```text

Productivity =

&#x20;   0.40 × Focus Score

&#x20; + 0.25 × Emotion Score

&#x20; + 0.25 × Attention Score

&#x20; + 0.10 × Posture Score

```



Focus and emotion labels are mapped into numerical values before the weighted score is calculated.



The final score is constrained to:



```text

0 – 100

```



This creates a unified session metric while retaining the individual behavioral signals underneath it.



\---



\# 📁 Project Structure



```text

COGNITO-AI-Focus-Monitoring-System/

│

├── models/

│   ├── cognito\_model.pkl

│   └── emotion\_model.pkl

│

├── modules/

│   ├── analytics\_engine.py

│   ├── break\_suggester.py

│   ├── confidence\_scorer.py

│   ├── dashboard.py

│   ├── distraction\_counter.py

│   ├── face\_recognition\_module.py

│   ├── multi\_person.py

│   ├── pdf\_export.py

│   ├── productivity\_score.py

│   ├── session\_logger.py

│   └── smart\_alerts.py

│

├── webapp/

│   ├── app.py

│   └── templates/

│       ├── admin.html

│       ├── dashboard.html

│       ├── index.html

│       ├── leaderboard.html

│       ├── login.html

│       ├── monitor.html

│       ├── onboarding.html

│       ├── profile.html

│       └── report.html

│

├── cognito\_final.py

├── cognito\_v2.py

├── cognito\_v3.py

├── realtime.py

│

├── extract\_features.py

├── extract\_frames.py

├── explore\_labels.py

│

├── train\_model.py

├── train\_emotion\_model.py

│

├── feedback.py

├── generate\_report.py

│

├── test\_dashboard.py

├── test\_distraction.py

├── test\_face\_recognition.py

├── test\_logger.py

├── test\_multi.py

├── test\_pdf.py

└── test\_productivity.py

```



\---



\# 🛠️ Technology Stack



\### Programming



\* Python



\### Computer Vision



\* OpenCV

\* MediaPipe Face Landmarker

\* MediaPipe Pose Landmarker

\* DeepFace



\### Machine Learning



\* Scikit-learn

\* Random Forest

\* SMOTE

\* Joblib

\* Pandas

\* NumPy



\### Web Development



\* Flask

\* Jinja2

\* HTML

\* CSS

\* JavaScript



\### Data \& Visualization



\* Pandas

\* Matplotlib

\* CSV

\* JSON



\### Desktop Interaction



\* Tkinter



\---



\# ⚙️ Installation



\## 1. Clone the repository



```bash

git clone https://github.com/sriharshini13/COGNITO-AI-Focus-Monitoring-System.git

cd COGNITO-AI-Focus-Monitoring-System

```



\## 2. Create a virtual environment



\### Windows



```bash

python -m venv venv

venv\\Scripts\\activate

```



\### macOS / Linux



```bash

python3 -m venv venv

source venv/bin/activate

```



\## 3. Install dependencies



The project uses packages including:



```bash

pip install flask

pip install opencv-python

pip install numpy

pip install pandas

pip install matplotlib

pip install scikit-learn

pip install imbalanced-learn

pip install joblib

pip install mediapipe

pip install deepface

```



> \*\*Note:\*\* Dependency versions may need adjustment depending on the local Python version and MediaPipe/DeepFace compatibility.



\---



\# ▶️ Running COGNITO



\## Web Application



From the project root:



```bash

python webapp/app.py

```



Then open the local Flask address shown in the terminal.



The application uses the system webcam for real-time monitoring.



\---



\## Standalone Real-Time Monitor



For the lightweight OpenCV monitoring pipeline:



```bash

python realtime.py

```



The standalone implementation loads:



```text

models/cognito\_model.pkl

face\_landmarker.task

```



and performs real-time focus classification from webcam frames.



Press:



```text

Q

```



to exit the monitoring window.



\---



\# 🧪 Model Training



The repository contains the training pipelines used to generate the saved ML models.



\### Train Focus Model



```bash

python train\_model.py

```



This expects the generated feature dataset:



```text

features/features.csv

```



and produces:



```text

models/cognito\_model.pkl

```



\### Train Emotion Model



```bash

python train\_emotion\_model.py

```



The emotion-training pipeline uses the feature data together with the corresponding label information and produces:



```text

models/emotion\_model.pkl

```



> The raw/generated dataset and feature directories are intentionally excluded from the public repository.



\---



\# 🧪 Testing



COGNITO includes separate test scripts covering important system components:



```text

test\_dashboard.py

test\_distraction.py

test\_face\_recognition.py

test\_logger.py

test\_multi.py

test\_pdf.py

test\_productivity.py

```



The tests cover areas such as:



\* Dashboard behavior

\* Distraction counting

\* Face recognition

\* Session logging

\* Multi-person functionality

\* PDF/report generation

\* Productivity scoring



\---



\# 🔐 Privacy \& Data Handling



COGNITO is designed around local webcam processing for its monitoring workflow.



The application stores session-related information locally, including:



\* User information

\* Session history

\* Productivity statistics

\* Focus statistics

\* Emotion statistics

\* Distraction information



The generated runtime directories are excluded from version control.



These include:



```text

dataset/

frames/

registered\_faces/

session\_logs/

reports/

features/

```



Users should review and configure local data storage appropriately before using the system with real personal data.



\---



\# ⚠️ Security Note



This repository is primarily a \*\*student/research/portfolio implementation\*\* and should not be treated as production-ready security infrastructure.



Before deploying publicly, review and replace development configuration such as:



\* Flask secret keys

\* Authentication configuration

\* Local JSON-based user storage

\* Webcam/data storage

\* Administrative access controls

\* Session management



In particular, production deployments should use environment variables for secrets rather than hard-coded application secrets.



\---



\# 📌 Current Implementation Status



\### Implemented



\* \[x] Real-time webcam processing

\* \[x] MediaPipe facial landmark extraction

\* \[x] EAR / MAR feature extraction

\* \[x] Gaze estimation

\* \[x] Head-pose feature extraction

\* \[x] Random Forest focus classification

\* \[x] Random Forest emotion classification

\* \[x] SMOTE-based training pipeline

\* \[x] Confidence estimation

\* \[x] Drowsiness detection

\* \[x] Yawning detection

\* \[x] Attention scoring

\* \[x] Productivity scoring

\* \[x] Distraction tracking

\* \[x] Focus streak tracking

\* \[x] Smart alerts

\* \[x] Break suggestions

\* \[x] Session logging

\* \[x] Session analytics

\* \[x] Flask web application

\* \[x] User authentication

\* \[x] User profiles

\* \[x] Multiple monitoring profiles

\* \[x] Dashboard

\* \[x] Reports

\* \[x] CSV export

\* \[x] Focus quizzes

\* \[x] Gamification / points

\* \[x] Leaderboard

\* \[x] Admin dashboard

\* \[x] Automated testing modules



\---



\# 🚧 Future Improvements



Potential next steps include:



\* \[ ] Add a formal `requirements.txt` with tested dependency versions

\* \[ ] Replace local JSON storage with a production database

\* \[ ] Move secrets to environment variables

\* \[ ] Improve model validation with subject-independent evaluation

\* \[ ] Add cross-validation and systematic hyperparameter tuning

\* \[ ] Add formal precision/recall/F1 benchmarking to the README

\* \[ ] Improve robustness under different lighting and camera conditions

\* \[ ] Improve calibration of model confidence scores

\* \[ ] Add better temporal modeling instead of frame-level predictions alone

\* \[ ] Introduce stronger temporal smoothing for focus/emotion predictions

\* \[ ] Add automated CI testing

\* \[ ] Containerize the application

\* \[ ] Deploy a production-ready version



\---



\# 🎯 Why COGNITO?



Traditional productivity applications usually depend on timers, manually entered tasks, or self-reported focus.



COGNITO explores a different approach:



```text

Observe

&#x20;  ↓

Extract behavioral signals

&#x20;  ↓

Classify

&#x20;  ↓

Interpret

&#x20;  ↓

Score

&#x20;  ↓

Alert

&#x20;  ↓

Analyze

&#x20;  ↓

Improve

```



Instead of simply telling a user to "focus for 25 minutes", COGNITO attempts to understand observable behavioral signals during the session and turn them into interpretable feedback.



\---



\# 💡 Technical Highlights



COGNITO demonstrates practical integration of several AI/software-engineering concepts:



\* Computer vision

\* Facial landmark processing

\* Feature engineering

\* Supervised machine learning

\* Class-imbalance handling

\* Model serialization

\* Real-time inference

\* Confidence estimation

\* Behavioral analytics

\* State-based event tracking

\* Flask backend development

\* Authentication

\* Persistent session storage

\* Dashboard development

\* Data export

\* Automated testing



The project therefore combines \*\*ML experimentation with a complete application layer\*\*, rather than stopping at model training.



\---



\# 👩‍💻 Author



\*\*Sri Harshini Pagadala\*\*



B.Tech — Computer Science \& Engineering (AI \& ML)



GitHub: \[@sriharshini13](https://github.com/sriharshini13)



LinkedIn: \[Sri Harshini Pagadala](https://www.linkedin.com/in/sri-harshini-pagadala/)



\---



\# 📄 License



This project is currently intended for educational, academic, and portfolio purposes.



A formal open-source license can be added when the project is ready for external contribution or redistribution.



