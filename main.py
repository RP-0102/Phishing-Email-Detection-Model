import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_pipeline():
    print("==================================================")
    print("       PHISHING EMAIL DETECTION ML ENGINE         ")
    print("==================================================")

    raw_data = {
        'email_text': [
            "URGENT: Your bank account has been locked. Click here to verify your identity immediately!",
            "Hey, are we still meeting up for lunch today at 12:30?",
            "Congratulations! You won a $1000 Walmart gift card. Claim your reward now by opening this URL.",
            "Attached is the project overview presentation for Q3 review. Let me know if you have questions.",
            "SYSTEM WARNING: Security alert for your account. Please reset your password using the link below.",
            "Don't forget that the monthly billing report is due this coming Friday afternoon.",
            "Your Netflix subscription payment declined. Update your billing credentials now to avoid service loss."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1]  # 1 = Phishing, 0 = Safe (Ham)
    }
    
    df = pd.DataFrame(raw_data)
    print(f"[*] Successfully ingested data sample ({len(df)} records loaded).")

    X = df['email_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("[*] Completed text tokenization and TF-IDF matrix transformations.")

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("[*] Classifier model training phase completed.")

    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n==================================================")
    print("             MODEL PERFORMANCE REPORT             ")
    print("==================================================")
    print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\n--- Confusion Matrix Matrix ---")
    print(f"True Negatives (Safe correctly flagged) : {cm[0][0]}")
    print(f"False Positives (Safe wrongly flagged) : {cm[0][1]}")
    print(f"False Negatives (Phishing missed!)     : {cm[1][0]}")
    print(f"True Positives (Phishing caught!)      : {cm[1][1]}")
    print("--------------------------------")

    print("\n[ Live Inference Simulation ]")
    live_test_email = ["Urgent notification: secure your account access now by clicking this web link."]
    live_tfidf = vectorizer.transform(live_test_email)
    prediction = model.predict(live_tfidf)
    
    status = "🚨 PHISHING DETECTED" if prediction[0] == 1 else "✅ SAFE EMAIL"
    print(f"Input Text : '{live_test_email[0]}'")
    print(f"Analysis Result : {status}")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()