def classify_softmax(text: str, vectorizer, model, label_encoder):
    # Preprocess
    X = vectorizer.transform([text]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        pred_class = logits.argmax(dim=1).item()

    return label_encoder.inverse_transform([pred_class])[0]
