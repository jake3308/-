    import time
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from torch.optim import AdamW

    # 데이터셋 로드 및 준비
    def load_and_prepare_dataset():
        dataset = load_dataset("stanfordnlp/sst2")

        # Tokenizer 로드
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 데이터 토큰화 함수
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

        # 데이터셋 토큰화
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return tokenized_datasets, tokenizer

    # 모델 정의
    class SentimentClassifier(nn.Module):
        def __init__(self):
            super(SentimentClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(outputs.pooler_output)
            return logits

    # 검증 정확도 계산 함수
    def calculate_accuracy(model, val_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        return accuracy

    # 모델 학습
    def train_model(model, train_loader, val_loader, optimizer, device):
        model.train()
        for epoch in range(3):  # 에포크 3회
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # 진행 상태 출력
                print(f"Processing batch {batch_idx}/{len(train_loader)}")
                
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

                # Batch 진행 상황 출력
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # 검증 정확도 계산
            val_accuracy = calculate_accuracy(model, val_loader, device)
            print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 모델 평가 및 정확도 계산
    def evaluate_model(model, tokenizer, device):
        test_sentences = [
            "it’s a charming and often affecting journey.",
            "unflinchingly bleak and desperate",
            "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker.",
            "the acting, costumes, music, cinematography and sound are all astounding given the production’s austere locales.",
            "it’s slow – very, very slow.",
            "although laced with humor and a few fanciful touches, the film is a refreshingly serious look at young women.",
            "a sometimes tedious film.",
            "or doing last year’s taxes with your ex-wife.",
            "you don’t have to know about music to appreciate the film’s easygoing blend of comedy and romance.",
            "in exactly 89 minutes, most of which passed as slowly as if i’d been sitting naked on an igloo, formula 51 sank from quirky to jerky to utter turkey."
        ]

        true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # 제공된 문장의 실제 레이블
        correct = 0

        # 테스트 문장 토큰화
        test_encodings = tokenizer(test_sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = test_encodings["input_ids"].to(device)
        attention_mask = test_encodings["attention_mask"].to(device)

        # 예측
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1)
        
        # 결과 출력 및 정확도 계산
        for i, sentence in enumerate(test_sentences):
            sentiment = "Positive" if predictions[i].item() == 1 else "Negative"
            correct += (predictions[i].item() == true_labels[i])
            print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")

        accuracy = correct / len(test_sentences)
        print(f"Test Accuracy: {accuracy:.4f}")

    # 메인 함수
    def main():
        # GPU 사용 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 데이터셋 준비
        start_time = time.time()  # 데이터 로드 시간 측정 시작
        tokenized_datasets, tokenizer = load_and_prepare_dataset()
        print(f"Dataset loading and preparation time: {time.time() - start_time:.2f} seconds")

        # 데이터셋 크기 확인
        print(f"Train dataset size: {len(tokenized_datasets['train'])}")
        print(f"Validation dataset size: {len(tokenized_datasets['validation'])}")

        # 학습 속도를 위해 샘플 데이터로 제한
        small_train_dataset = tokenized_datasets["train"].select(range(1000))  # 1000개 샘플로 축소
        train_loader = DataLoader(small_train_dataset, shuffle=True, batch_size=8, num_workers=4)

        small_val_dataset = tokenized_datasets["validation"].select(range(200))  # 200개로 축소
        val_loader = DataLoader(small_val_dataset, batch_size=8, num_workers=4)

        # 모델 초기화
        model = SentimentClassifier().to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # 학습
        print("Training the model...")
        train_model(model, train_loader, val_loader, optimizer, device)

        # 테스트
        print("\nEvaluating the model on test sentences...")
        evaluate_model(model, tokenizer, device)

    if __name__ == "__main__":
        main()
