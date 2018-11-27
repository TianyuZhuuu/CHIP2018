from _00_imports import *
from _06_training_utils import fit

warnings.filterwarnings('ignore')


class BaseModel(nn.Module):
    @staticmethod
    def interact(x1, x2, dim=-1):
        """

        :param x1: [batch, (len1), dim]
        :param x2: [batch, (len2), dim]
        :return:
        """
        diff = torch.abs(x1 - x2)
        prod = x1 * x2
        concat = torch.cat((x1, x2, diff, prod), dim=dim)
        return concat

    @staticmethod
    def avg_max_pool(x):
        """

        :param x: [batch, len, dim]
        :return: [batch, 2*dim]
        """
        p1 = x.mean(dim=1)
        p2 = x.max(dim=1)[0]
        concat = torch.cat((p1, p2), dim=-1)  # [batch, 2*dim]
        return concat

    @staticmethod
    def soft_attention_align(x1, x2, mask1, mask2):
        """

        :param x1: [batch, len1, dim]
        :param x2: [batch, len2, dim]
        :param mask1: [batch, len1]
        :param mask2: [batch, len2]
        :return: x1_align, x2_align
        """
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        score = torch.bmm(x1, x2.transpose(1, 2))  # [batch, len1, len2]
        weight1 = F.softmax(score + mask2.unsqueeze(1), dim=-1)  # [batch, len1, len2]
        x1_align = torch.bmm(weight1, x2)  # [batch, len1, dim]
        weight2 = F.softmax(score.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)  # [batch, len2, len1]
        x2_align = torch.bmm(weight2, x1)
        return x1_align, x2_align

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def dropout(self, x, p):
        return F.dropout(x, p=p, training=self.training)

    def a(self):
        pass


class InferSent(BaseModel):
    def __init__(self, word_embedding, char_embedding, encode_dim, fc_dim1, fc_dim2, use_feature=False):
        super(InferSent, self).__init__()
        self.use_feature = use_feature
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding, freeze=False),
            nn.Dropout(0.2)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding, freeze=False),
            nn.Dropout(0.2)
        )
        embed_dim = word_embedding.size(1)
        self.word_lstm = nn.LSTM(embed_dim, encode_dim, batch_first=True, bidirectional=True)
        self.char_lstm = nn.LSTM(embed_dim, encode_dim, batch_first=True, bidirectional=True)

        fc_input_dim = 16 * encode_dim
        if use_feature:
            fc_input_dim += 100
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_dim, fc_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim1, fc_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.out = nn.Linear(fc_dim2, 1)
        self.initialize_weights()

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        word_e1 = self.word_embed(q1_word)
        word_e2 = self.word_embed(q2_word)
        char_e1 = self.char_embed(q1_char)
        char_e2 = self.char_embed(q2_char)
        word_o1 = self.word_lstm(word_e1)[0]
        word_o2 = self.word_lstm(word_e2)[0]
        char_o1 = self.char_lstm(char_e1)[0]
        char_o2 = self.char_lstm(char_e2)[0]
        word_v1 = word_o1.max(dim=1)[0]
        word_v2 = word_o2.max(dim=1)[0]
        char_v1 = char_o1.max(dim=1)[0]
        char_v2 = char_o2.max(dim=1)[0]
        word_interaction = self.interact(word_v1, word_v2)
        char_interaction = self.interact(char_v1, char_v2)
        interaction = torch.cat((word_interaction, char_interaction), dim=1)
        out = F.dropout(interaction, p=0.5, training=self.training)
        if self.use_feature:
            feat = F.dropout(feature, p=0.3, training=self.training)
            out = torch.cat((out, feat), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.out(out)
        return F.sigmoid(out)


class SiameseTextCNN(BaseModel):
    def __init__(self, word_embedding, char_embedding, filter_num, fc_dim, use_feature=False):
        super(SiameseTextCNN, self).__init__()
        self.use_feature = use_feature
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding, freeze=False),
            nn.Dropout(p=0.2)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding, freeze=False),
            nn.Dropout(p=0.2)
        )
        self.word_branch1 = nn.Sequential(
            nn.Conv2d(1, filter_num, (1, 300)),
            nn.ReLU(inplace=True)
        )
        self.word_branch2 = nn.Sequential(
            nn.Conv2d(1, filter_num, (2, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.word_branch3 = nn.Sequential(
            nn.Conv2d(1, filter_num, (3, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.char_branch1 = nn.Sequential(
            nn.Conv2d(1, filter_num, (1, 300)),
            nn.ReLU(inplace=True)
        )
        self.char_branch2 = nn.Sequential(
            nn.Conv2d(1, filter_num, (2, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.char_branch3 = nn.Sequential(
            nn.Conv2d(1, filter_num, (3, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )

        fc_input_dim = 24 * filter_num
        if use_feature:
            fc_input_dim += 100
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.out = nn.Linear(fc_dim, 1)
        self.initialize_weights()

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        q1_word_embed = self.word_embed(q1_word).unsqueeze(dim=1)  # [batch, 6, seq_len, 300]
        q2_word_embed = self.word_embed(q2_word).unsqueeze(dim=1)
        q1_char_embed = self.char_embed(q1_char).unsqueeze(dim=1)
        q2_char_embed = self.char_embed(q2_char).unsqueeze(dim=1)

        # conv out: [batch, num_filter, seq_len-height+6, 6]
        # squeeze & max: [batch, num_filter]
        q1_word_branch1, _ = self.word_branch1(q1_word_embed).squeeze(dim=-1).max(dim=-1)
        q1_word_branch2, _ = self.word_branch2(q1_word_embed).squeeze(dim=-1).max(dim=-1)
        q1_word_branch3, _ = self.word_branch3(q1_word_embed).squeeze(dim=-1).max(dim=-1)
        q2_word_branch1, _ = self.word_branch1(q2_word_embed).squeeze(dim=-1).max(dim=-1)
        q2_word_branch2, _ = self.word_branch2(q2_word_embed).squeeze(dim=-1).max(dim=-1)
        q2_word_branch3, _ = self.word_branch3(q2_word_embed).squeeze(dim=-1).max(dim=-1)

        q1_char_branch1, _ = self.char_branch1(q1_char_embed).squeeze(dim=-1).max(dim=-1)
        q1_char_branch2, _ = self.char_branch2(q1_char_embed).squeeze(dim=-1).max(dim=-1)
        q1_char_branch3, _ = self.char_branch3(q1_char_embed).squeeze(dim=-1).max(dim=-1)
        q2_char_branch1, _ = self.char_branch1(q2_char_embed).squeeze(dim=-1).max(dim=-1)
        q2_char_branch2, _ = self.char_branch2(q2_char_embed).squeeze(dim=-1).max(dim=-1)
        q2_char_branch3, _ = self.char_branch3(q2_char_embed).squeeze(dim=-1).max(dim=-1)

        q1_word_out = torch.cat((q1_word_branch1, q1_word_branch2, q1_word_branch3), dim=1)
        q2_word_out = torch.cat((q2_word_branch1, q2_word_branch2, q2_word_branch3), dim=1)
        q1_char_out = torch.cat((q1_char_branch1, q1_char_branch2, q1_char_branch3), dim=1)
        q2_char_out = torch.cat((q2_char_branch1, q2_char_branch2, q2_char_branch3), dim=1)

        word_repre = self.interact(q1_word_out, q2_word_out)  # [batch, 3*filter_num]
        char_repre = self.interact(q1_char_out, q2_char_out)  # [batch, 3*filter_num]

        if not self.use_feature:
            repre = torch.cat((word_repre, char_repre), dim=1)
        else:
            repre = torch.cat((word_repre, char_repre, feature), dim=1)
        repre = self.dropout(repre, p=0.5)

        out = self.fc(repre)
        out = self.out(out)
        return F.sigmoid(out)


class DecomposableAttention(BaseModel):
    def __init__(self, word_embedding, char_embedding, attend_dim, compare_dim, fc_dim, use_feature=False):
        super(DecomposableAttention, self).__init__()
        self.use_feature = use_feature
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding),
            nn.Dropout(p=0.2)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding),
            nn.Dropout(p=0.2)
        )

        self.word_attend_fc = nn.Sequential(
            nn.Linear(300, attend_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.char_attend_fc = nn.Sequential(
            nn.Linear(300, attend_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.word_compare_fc = nn.Sequential(
            nn.Linear(600, compare_dim),
            nn.ReLU(inplace=True),
        )
        self.char_compare_fc = nn.Sequential(
            nn.Linear(600, compare_dim),
            nn.ReLU(inplace=True),
        )

        fc_input_dim = 16 * compare_dim
        if use_feature:
            fc_input_dim += 100
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.out = nn.Linear(fc_dim, 1)
        self.initialize_weights()

    def alignment(self, e1, e2, a1, a2, mask1, mask2):
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))  # [batch, len1]
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))  # [batch, len2]
        score = torch.bmm(a1, a2.transpose(1, 2))  # [batch, len1, len2]
        weight1 = F.softmax(score + mask2.unsqueeze(1), dim=-1)
        align1 = torch.bmm(weight1, e2)  # [batch, len1, dim]
        weight2 = F.softmax(score.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)  # [batch, len2, dim]
        align2 = torch.bmm(weight2, e1)
        return align1, align2

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        word_mask1, word_mask2 = q1_word.eq(1), q2_word.eq(1)
        char_mask1, char_mask2 = q1_char.eq(1), q2_char.eq(1)
        word_a = self.word_embed(q1_word)  # [batch, len_1, 300]
        word_b = self.word_embed(q2_word)  # [batch, len_2, 300]
        char_a = self.char_embed(q1_char)
        char_b = self.char_embed(q2_char)

        # Attend
        word_a_attend = self.word_attend_fc(word_a)  # [batch, len_1, attend_dim]
        word_b_attend = self.word_attend_fc(word_b)  # [batch, len_2, attend_dim]
        word_beta, word_alpha = self.alignment(word_a, word_b, word_a_attend, word_b_attend, word_mask1, word_mask2)

        char_a_attend = self.char_attend_fc(char_a)
        char_b_attend = self.char_attend_fc(char_b)
        char_beta, char_alpha = self.alignment(char_a, char_b, char_a_attend, char_b_attend, char_mask1, char_mask2)

        # Compare
        word_a_beta = torch.cat((word_a, word_beta), dim=2)
        word_b_alpha = torch.cat((word_b, word_alpha), dim=2)
        word_v1 = self.word_compare_fc(word_a_beta)  # [batch, len_1, compare_dim]
        word_v2 = self.word_compare_fc(word_b_alpha)  # [batch, len_2, compare_dim]

        char_a_beta = torch.cat((char_a, char_beta), dim=2)
        char_b_alpha = torch.cat((char_b, char_alpha), dim=2)
        char_v1 = self.char_compare_fc(char_a_beta)
        char_v2 = self.char_compare_fc(char_b_alpha)

        # Aggregate
        word_p1 = self.avg_max_pool(word_v1)
        word_p2 = self.avg_max_pool(word_v2)
        char_p1 = self.avg_max_pool(char_v1)
        char_p2 = self.avg_max_pool(char_v2)

        word_interaction = self.interact(word_p1, word_p2)
        char_interaction = self.interact(char_p1, char_p2)

        # [batch, 16 * compare_dim]
        if not self.use_feature:
            v_concat = torch.cat((word_interaction, char_interaction), dim=1)
        else:
            v_concat = torch.cat((word_interaction, char_interaction, feature), dim=1)
        v_concat = self.dropout(v_concat, p=0.5)
        out = self.fc(v_concat)
        out = self.out(out)
        return F.sigmoid(out)


if __name__ == '__main__':
    word_vector_path = '../data/word_vectors.txt'
    char_vector_path = '../data/char_vectors.txt'

    # CV: 0.8572 LB: 0.85251
    model_fn = lambda word_embedding, char_embedding: InferSent(word_embedding, char_embedding, 64, 256, 128)
    for seed in range(1, 11):
        fit(model_fn, 'InferSent', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8642 LB: ?
    model_fn = lambda word_embedding, char_embedding: InferSent(word_embedding, char_embedding, 64, 256, 128, True)
    for seed in range(1, 11):
        fit(model_fn, 'InferSent_Feat', word_vector_path, char_vector_path, 10, seed=seed)

    # CV: 0.8531 LB: ?
    model_fn = lambda word_embedding, char_embedding: SiameseTextCNN(word_embedding, char_embedding, 64, 128)
    for seed in range(1, 11):
        fit(model_fn, 'SiameseTextCNN_1124', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8570 LB: ?
    model_fn = lambda word_embedding, char_embedding: SiameseTextCNN(word_embedding, char_embedding, 64, 128, True)
    for seed in range(1, 11):
        fit(model_fn, 'SiameseTextCNN_Feat_1124', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8527 LB: ?
    model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64,
                                                                            128)
    for seed in range(1, 11):
        fit(model_fn, 'DecomposableAttention_1125', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8622 LB: ?
    model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64, 128,
                                                                            True)
    for seed in range(1, 11):
        fit(model_fn, 'DecomposableAttention_Feat_1125', word_vector_path, char_vector_path, num_folds=10, seed=seed)
