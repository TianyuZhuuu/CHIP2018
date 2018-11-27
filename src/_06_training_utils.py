from sklearn.metrics import log_loss

from _00_imports import *
from _05_prepare_data import CHIPDataset


def fit(model_fn, model_name, word_path, char_path, num_folds=5, batch_size=32, lr=1e-3, seed=1):
    qid_path = '../input/question_id.csv'
    train_path = '../input/train.csv'
    test_path = '../input/test.csv'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = CHIPDataset(qid_path, train_path, test_path, word_path, char_path, num_folds=num_folds,
                          batch_size=batch_size)
    word_embedding = dataset.word_embedding
    char_embedding = dataset.char_embedding

    train_feat = torch.tensor(pd.read_csv('../data/simplified_train_feat.csv').values, dtype=torch.float).to(device)
    test_feat = torch.tensor(pd.read_csv('../data/simplified_test_feat.csv').values, dtype=torch.float).to(device)

    N_EPOCHS = 10000
    PATIENCE = 10
    CHECKPOINT = f'../checkpoint/{model_name}/{seed}/'
    OOF_DIR = f'../oof/{model_name}/{seed}/'
    if not os.path.exists(CHECKPOINT):
        os.makedirs(CHECKPOINT)
    if not os.path.exists(OOF_DIR):
        os.makedirs(OOF_DIR)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    oof_train_pred_f1, oof_test_pred_f1 = [], []
    oof_train_pred_loss, oof_test_pred_loss = [], []

    for fold in range(num_folds):
        print(f'Processing fold {fold}')
        train_iter, valid_iter, test_iter = dataset.generate_iterator_fold(fold)
        model = model_fn(word_embedding, char_embedding).to(device)

        BEST_EPOCH = 0
        BEST_F1 = -1
        BEST_LOSS = 1e3
        BEST_LOSS_CHECKPOINT_TEMPLATE = CHECKPOINT + 'fold{}_loss_{:.4f}.ckpt'
        BEST_F1_CHECKPOINT_TEMPLATE = CHECKPOINT + 'fold{}_f1_{:.4f}.ckpt'
        BEST_F1_CHECKPOINT = None
        BEST_LOSS_CHECKPOINT = None

        criterion = nn.BCELoss(size_average=False)
        trainable_params = [param for name, param in model.named_parameters() if 'embed' not in name]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

        for epoch in range(N_EPOCHS):

            if epoch - BEST_EPOCH > PATIENCE:
                print(f'No improvement for {PATIENCE} epochs, stop training...')
                break

            scheduler.step(epoch)

            start = time.time()
            model.train()
            train_label, train_pred_f1 = [], []
            train_loss = 0.0
            for data in train_iter:
                index, q1_word, q2_word, q1_char, q2_char, label = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char, data.label
                feat = train_feat[index, :]

                prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                loss = criterion(prob, label)

                train_loss += loss.item()

                train_label.append(label)
                train_pred_f1.append(prob)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_label = torch.cat(train_label, 0).cpu().detach().numpy()
            train_pred_f1 = torch.cat(train_pred_f1, 0).cpu().detach().numpy()

            train_f1 = f1_score(train_label, np.round(train_pred_f1))
            train_loss /= len(train_label)

            model.eval()
            valid_label, valid_pred = [], []
            valid_loss = 0.0
            with torch.no_grad():
                for data in valid_iter:
                    index, q1_word, q2_word, q1_char, q2_char, label = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char, data.label
                    feat = train_feat[index, :]

                    prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                    loss = criterion(prob, label)
                    valid_loss += loss.item()

                    valid_label.append(label)
                    valid_pred.append(prob)

            valid_label = torch.cat(valid_label, 0).cpu().detach().numpy()
            valid_pred = torch.cat(valid_pred, 0).cpu().detach().numpy()

            valid_f1 = f1_score(valid_label, np.round(valid_pred))
            valid_loss /= len(valid_label)

            if valid_f1 > BEST_F1:
                BEST_EPOCH = epoch
                BEST_F1 = valid_f1

                if BEST_F1_CHECKPOINT is not None and os.path.exists(BEST_F1_CHECKPOINT):
                    os.remove(BEST_F1_CHECKPOINT)

                BEST_F1_CHECKPOINT = BEST_F1_CHECKPOINT_TEMPLATE.format(fold, valid_f1)
                torch.save(model.state_dict(), BEST_F1_CHECKPOINT)

            if valid_loss < BEST_LOSS:
                BEST_EPOCH = epoch
                BEST_LOSS = valid_loss

                if BEST_LOSS_CHECKPOINT is not None and os.path.exists(BEST_LOSS_CHECKPOINT):
                    os.remove(BEST_LOSS_CHECKPOINT)

                BEST_LOSS_CHECKPOINT = BEST_LOSS_CHECKPOINT_TEMPLATE.format(fold, valid_loss)
                torch.save(model.state_dict(), BEST_LOSS_CHECKPOINT)

            end = time.time()
            elapsed = end - start

            print(f'Epoch {epoch}: {elapsed:.1f}s')
            print(f'\ttrain_loss: {train_loss:.4f}\ttrain_f1: {train_f1:.4f}')
            print(f'\tvalid_loss: {valid_loss:.4f}\tvalid_f1: {valid_f1:.4f}')
            print('=' * 50)

        model.load_state_dict(torch.load(BEST_F1_CHECKPOINT))
        model.eval()
        _train_pred, _test_pred = [], []
        with torch.no_grad():
            for data in valid_iter:
                index, q1_word, q2_word, q1_char, q2_char = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char
                feat = train_feat[index, :]
                prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                _train_pred.append(prob)
        _train_pred = np.concatenate(_train_pred)

        model.eval()
        with torch.no_grad():
            for data in test_iter:
                index, q1_word, q2_word, q1_char, q2_char = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char
                feat = test_feat[index, :]
                prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                _test_pred.append(prob)
        _test_pred = np.concatenate(_test_pred)

        oof_train_pred_f1.append(_train_pred)
        oof_test_pred_f1.append(_test_pred)

        model.load_state_dict(torch.load(BEST_LOSS_CHECKPOINT))
        model.eval()
        _train_pred, _test_pred = [], []
        with torch.no_grad():
            for data in valid_iter:
                index, q1_word, q2_word, q1_char, q2_char = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char
                feat = train_feat[index, :]
                prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                _train_pred.append(prob)

        _train_pred = np.concatenate(_train_pred)

        with torch.no_grad():
            for data in test_iter:
                index, q1_word, q2_word, q1_char, q2_char = data.index, data.q1_word, data.q2_word, data.q1_char, data.q2_char
                feat = test_feat[index, :]
                prob = model(q1_word, q2_word, q1_char, q2_char, feat).squeeze()
                _test_pred.append(prob)
        _test_pred = np.concatenate(_test_pred)

        oof_train_pred_loss.append(_train_pred)
        oof_test_pred_loss.append(_test_pred)

    train_pred_f1 = dataset.reorder_oof_prediction(np.concatenate(oof_train_pred_f1))
    f1 = f1_score(truth, np.round(train_pred_f1))
    loss = log_loss(truth, np.stack((1 - train_pred_f1, train_pred_f1), axis=1))
    train_path = OOF_DIR + f'train_f1_{f1:.4f}_{loss:.4f}'
    np.save(train_path, train_pred_f1)

    test_pred = np.array(oof_test_pred_f1).mean(axis=0)
    test_path = OOF_DIR + f'test_f1_{f1:.4f}_{loss:.4f}'
    np.save(test_path, test_pred)

    train_pred_loss = dataset.reorder_oof_prediction(np.concatenate(oof_train_pred_loss))
    f1 = f1_score(truth, np.round(train_pred_loss))
    loss = log_loss(truth, np.stack((1 - train_pred_loss, train_pred_loss), axis=1))
    train_path = OOF_DIR + f'train_loss_{f1:.4f}_{loss:.4f}'
    np.save(train_path, train_pred_loss)

    test_pred = np.array(oof_test_pred_loss).mean(axis=0)
    test_path = OOF_DIR + f'test_loss_{f1:.4f}_{loss:.4f}'
    np.save(test_path, test_pred)