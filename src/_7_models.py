from _0_imports import *
from _6_training_utils import fit, finetune_embedding

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
    def __init__(self, word_embedding, char_embedding, encode_dim, fc_dim1, fc_dim2):
        super(InferSent, self).__init__()
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
        self.fc1 = nn.Sequential(
            nn.Linear(16 * encode_dim, fc_dim1),
            nn.ReLU(),
            nn.Dropout(),
            # nn.BatchNorm1d(fc_dim1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim1, fc_dim2),
            nn.ReLU(),
            nn.Dropout(),
            # nn.BatchNorm1d(fc_dim2)
        )
        self.out = nn.Linear(fc_dim2, 1)
        self.initialize_weights()

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, _ = input
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
        interaction = F.dropout(interaction, p=0.5, training=self.training)
        out = self.fc1(interaction)
        out = self.fc2(out)
        out = self.out(out)
        return F.sigmoid(out)


class LexicalDecomposition(BaseModel):
    def __init__(self, word_embedding, char_embedding, filter_num):
        super(LexicalDecomposition, self).__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding, freeze=False),
            nn.Dropout(0.2)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding, freeze=False),
            nn.Dropout(0.2)
        )
        self.word_branch1 = nn.Sequential(
            nn.Conv2d(2, filter_num, (1, 300)),
            nn.ReLU(inplace=True)
        )
        self.word_branch2 = nn.Sequential(
            nn.Conv2d(2, filter_num, (2, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.word_branch3 = nn.Sequential(
            nn.Conv2d(2, filter_num, (3, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.char_branch1 = nn.Sequential(
            nn.Conv2d(2, filter_num, (1, 300)),
            nn.ReLU(inplace=True)
        )
        self.char_branch2 = nn.Sequential(
            nn.Conv2d(2, filter_num, (2, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.char_branch3 = nn.Sequential(
            nn.Conv2d(2, filter_num, (3, 300), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(12 * filter_num, 1)
        self.initialize_weights()

    def semantic_matching(self, x1, x2, mask1, mask2):
        """

        :param x1: [batch, len1, dim]
        :param x2: [batch, len2, dim]
        :param mask1: [batch, len1] 1 for padding
        :param mask2: [batch, len2]
        :return:
        """
        len1, len2 = x1.size(1), x2.size(1)
        x1_expand = x1.unsqueeze(2).repeat(1, 1, len2, 1)  # [batch, len1, len2, embed_dim]
        x2_expand = x2.unsqueeze(1).repeat(1, len1, 1, 1)
        alpha = F.cosine_similarity(x1_expand, x2_expand, dim=-1)  # [batch, len1, len2]
        alpha_mask1 = alpha * (~mask2).float().unsqueeze(1)
        alpha_mask2 = alpha * (~mask1).float().unsqueeze(2)
        x1_match = torch.bmm(alpha_mask1, x2)  # [batch, len1, dim]
        x2_match = torch.bmm(alpha_mask2.transpose(1, 2), x1)  # [batch, len2, dim]
        return x1_match, x2_match

    def linear_decomposition(self, x, x_match):
        """

        :param x: [batch, len, dim]
        :param x_match: [batch, len, dim]
        :return:
        """
        score = F.cosine_similarity(x, x_match, dim=-1).unsqueeze(-1)  # [batch, len, 1]
        x_sim = score * x
        x_dis = (1 - score) * x
        return x_sim, x_dis

    def composition(self, sim, dis, granularity='word', drop_rate=0.2):
        # print(sim.size())
        # print(dis.size())
        compound = torch.cat((sim.unsqueeze(1), dis.unsqueeze(1)), dim=1)  # [batch, 2, len, dim]
        if granularity == 'word':
            o1 = self.word_branch1(compound).squeeze().max(dim=-1)[0]  # [batch, filter_num]
            o2 = self.word_branch2(compound).squeeze().max(dim=-1)[0]
            o3 = self.word_branch3(compound).squeeze().max(dim=-1)[0]
        elif granularity == 'char':
            o1 = self.char_branch1(compound).squeeze().max(dim=-1)[0]  # [batch, filter_num]
            o2 = self.char_branch2(compound).squeeze().max(dim=-1)[0]
            o3 = self.char_branch3(compound).squeeze().max(dim=-1)[0]
        else:
            raise ValueError(f'Unknown granularity {granularity}')
        out = torch.cat((o1, o2, o3), dim=-1)  # [batch, 3*filter_num]
        if drop_rate != 0:
            out = self.dropout(out, p=drop_rate)
        return out

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        word_mask1, word_mask2 = q1_word.eq(1), q2_word.eq(1)
        char_mask1, char_mask2 = q1_char.eq(1), q2_char.eq(1)
        # Word Representation
        word_e1, word_e2 = self.word_embed(q1_word), self.word_embed(q2_word)  # [batch, seq_len, embed_dim]
        char_e1, char_e2 = self.char_embed(q1_char), self.char_embed(q2_char)
        # Semantic Matching
        word_match1, word_match2 = self.semantic_matching(word_e1, word_e2, word_mask1, word_mask2)
        char_match1, char_match2 = self.semantic_matching(char_e1, char_e2, char_mask1, char_mask2)
        # Decomposition
        word_sim1, word_dis1 = self.linear_decomposition(word_e1, word_match1)  # [batch, len, dim]
        word_sim2, word_dis2 = self.linear_decomposition(word_e2, word_match2)
        char_sim1, char_dis1 = self.linear_decomposition(char_e1, char_match1)
        char_sim2, char_dis2 = self.linear_decomposition(char_e2, char_match2)
        # Composition
        word_comp1 = self.composition(word_sim1, word_dis1, granularity='word')
        word_comp2 = self.composition(word_sim2, word_dis2, granularity='word')
        char_comp1 = self.composition(char_sim1, char_dis1, granularity='char')
        char_comp2 = self.composition(char_sim2, char_dis2, granularity='char')
        # Similarity assessing
        out = torch.cat((word_comp1, word_comp2, char_comp1, char_comp2), dim=-1)
        out = self.out(out)
        return F.sigmoid(out)


class InferSent_Feat(BaseModel):
    def __init__(self, word_embedding, char_embedding, encode_dim, fc_dim1, fc_dim2):
        super(InferSent_Feat, self).__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding),
            nn.Dropout(0.2)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding),
            nn.Dropout(0.2)
        )
        self.word_lstm = nn.LSTM(300, encode_dim, batch_first=True, bidirectional=True)
        self.char_lstm = nn.LSTM(300, encode_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * encode_dim + 100, fc_dim1),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim1, fc_dim2),
            nn.ReLU(),
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
        interaction = F.dropout(interaction, p=0.5, training=self.training)
        feat = F.dropout(feature, p=0.3, training=self.training)
        out = self.fc1(torch.cat((interaction, feat), dim=1))
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


class SiameseTextRNNCNN(BaseModel):
    def __init__(self, word_embedding, char_embedding, encode_dim, filter_num, fc_dim):
        super(SiameseTextRNNCNN, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding)
        self.char_embedding = nn.Embedding.from_pretrained(char_embedding)
        self.word_lstm = nn.LSTM(300, encode_dim, batch_first=True, bidirectional=True)
        self.char_lstm = nn.LSTM(300, encode_dim, batch_first=True, bidirectional=True)

        self.word_branch1 = nn.Conv2d(1, filter_num, (2, 2 * encode_dim), padding=(1, 0))
        self.word_branch2 = nn.Conv2d(1, filter_num, (3, 2 * encode_dim), padding=(1, 0))
        self.word_branch3 = nn.Conv2d(1, filter_num, (4, 2 * encode_dim), padding=(2, 0))
        self.word_branch4 = nn.Conv2d(1, filter_num, (5, 2 * encode_dim), padding=(2, 0))

        self.char_branch1 = nn.Conv2d(1, filter_num, (2, 2 * encode_dim), padding=(1, 0))
        self.char_branch2 = nn.Conv2d(1, filter_num, (3, 2 * encode_dim), padding=(1, 0))
        self.char_branch3 = nn.Conv2d(1, filter_num, (4, 2 * encode_dim), padding=(2, 0))
        self.char_branch4 = nn.Conv2d(1, filter_num, (5, 2 * encode_dim), padding=(2, 0))

        self.fc = nn.Sequential(
            nn.Linear(32 * filter_num, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(fc_dim, 1)
        self.initialize_weights()

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feat = input
        word_e1, word_e2 = self.word_embedding(q1_word), self.word_embedding(q2_word)
        char_e1, char_e2 = self.char_embedding(q1_char), self.char_embedding(q2_char)
        word_o1, word_o2 = self.word_lstm(word_e1)[0], self.word_lstm(word_e2)[0]
        char_o1, char_o2 = self.char_lstm(char_e1)[0], self.char_lstm(char_e2)[0]

        # [batch, 1, seq_len, dim]
        word_o1, word_o2 = self.dropout(word_o1.unsqueeze(1), p=0.4), self.dropout(word_o2.unsqueeze(1), p=0.4)
        char_o1, char_o2 = self.dropout(char_o1.unsqueeze(1), p=0.4), self.dropout(char_o2.unsqueeze(1), p=0.4)

        word_q1_b1 = self.word_branch1(word_o1).squeeze(dim=-1).max(dim=-1)[0]
        word_q1_b2 = self.word_branch2(word_o1).squeeze(dim=-1).max(dim=-1)[0]
        word_q1_b3 = self.word_branch3(word_o1).squeeze(dim=-1).max(dim=-1)[0]
        word_q1_b4 = self.word_branch4(word_o1).squeeze(dim=-1).max(dim=-1)[0]
        word_q2_b1 = self.word_branch1(word_o2).squeeze(dim=-1).max(dim=-1)[0]
        word_q2_b2 = self.word_branch2(word_o2).squeeze(dim=-1).max(dim=-1)[0]
        word_q2_b3 = self.word_branch3(word_o2).squeeze(dim=-1).max(dim=-1)[0]
        word_q2_b4 = self.word_branch4(word_o2).squeeze(dim=-1).max(dim=-1)[0]

        char_q1_b1 = self.char_branch1(char_o1).squeeze(dim=-1).max(dim=-1)[0]
        char_q1_b2 = self.char_branch2(char_o1).squeeze(dim=-1).max(dim=-1)[0]
        char_q1_b3 = self.char_branch3(char_o1).squeeze(dim=-1).max(dim=-1)[0]
        char_q1_b4 = self.char_branch4(char_o1).squeeze(dim=-1).max(dim=-1)[0]
        char_q2_b1 = self.char_branch1(char_o2).squeeze(dim=-1).max(dim=-1)[0]
        char_q2_b2 = self.char_branch2(char_o2).squeeze(dim=-1).max(dim=-1)[0]
        char_q2_b3 = self.char_branch3(char_o2).squeeze(dim=-1).max(dim=-1)[0]
        char_q2_b4 = self.char_branch4(char_o2).squeeze(dim=-1).max(dim=-1)[0]

        word_conv_o1 = torch.cat((word_q1_b1, word_q1_b2, word_q1_b3, word_q1_b4), dim=1)
        word_conv_o2 = torch.cat((word_q2_b1, word_q2_b2, word_q2_b3, word_q2_b4), dim=1)
        char_conv_o1 = torch.cat((char_q1_b1, char_q1_b2, char_q1_b3, char_q1_b4), dim=1)
        char_conv_o2 = torch.cat((char_q2_b1, char_q2_b2, char_q2_b3, char_q2_b4), dim=1)

        word_v = self.interact(word_conv_o1, word_conv_o2)
        char_v = self.interact(char_conv_o1, char_conv_o2)
        v = self.dropout(torch.cat((word_v, char_v), dim=-1), p=0.5)
        out = self.fc(v)
        out = self.out(out)
        return F.sigmoid(out)


class BiMPM(BaseModel):
    def __init__(self, word_embedding, char_embedding, encode_dim, num_perspective):
        super(BiMPM, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding)
        self.char_embedding = nn.Embedding.from_pretrained(char_embedding)
        self.word_lstm = nn.LSTM(300, encode_dim, bidirectional=True, batch_first=True)
        self.char_lstm = nn.LSTM(300, encode_dim, bidirectional=True, batch_first=True)
        self.num_perspective = num_perspective
        self.word_W = nn.Parameter(torch.tensor((2, 6, num_perspective, 2 * encode_dim)))

    # def cosine_match(self, p, q, W):
    #     """
    #
    #     :param p: [batch, dim]
    #     :param q: [batch, dim]
    #     :param W: [num_perspective, dim]
    #     :return: [batch, num_perspective]
    #     """
    #     p_expand = p.unsqueeze(1).repeat(1, self.num_perspective, 1) # [batch, num_perspective,

    def full_matching(self, p, q, W):
        """

        :param p: [batch, seq_len, dim]
        :param q: [batch, dim]
        :param W: [num_perspective, dim]
        :return:
        """
        p_proj = p.unsqueeze(2).repeat(1, 1, self.num_perspective, 1) * W  # [batch, seq_len, num_perspective, dim]
        q_proj = q.unsqueeze(1).unsqueeze(1).expand_as(p_proj) * W  # [batch, seq_len, num_perspective, dim]
        rt = F.cosine_similarity(p_proj, q_proj, dim=-1)  # [batch, seq_len, num_perspective]
        return rt

    def maxpooling_matching(self, p, q, W):
        """

        :param p: [batch, len1, dim]
        :param q: [batch, len2, dim]
        :param W: [num_perspective, dim]
        :return:
        """
        len1, len2 = p.size(1), q.size(1)
        # [batch, len1, len2, num_perspective, dim]
        p_proj = p.unsqueeze(2).unsqueeze(2).repeat(1, 1, len2, self.num_perspective, 1) * W
        q_proj = q.unsqueeze(1).unsqueeze(3).repeat(1, len1, 1, self.num_perspective, 1) * W
        match = F.cosine_similarity(p_proj, q_proj, dim=-1)  # [batch, len1, len2, num_perspective]
        rt = match.max(match, dim=2)[0]  # [batch, len1, num_perspective]
        return rt

    def attentive_matching(self, p, q, W):
        """

        :param p: [batch, len1, dim]
        :param q: [batch, len2, dim]
        :param W: [num_perspective, dim]
        :return:
        """
        len1, len2 = p.size(1), q.size(1)
        p_expand = p.unsqueeze(2).repeat(1, 1, len2, 1)  # [batch, len1, len2, dim]
        q_expand = q.unsqueeze(1).repeat(1, len1, 1, 1)  # [batch, len1, len2, dim]

        alpha = F.cosine_similarity(p_expand, q_expand, dim=-1)  # [batch, len1, len2]
        alpha_normalized = alpha / alpha.sum(dim=2, keepdim=True)  # [batch, len1, len2]

        p_align = torch.bmm(alpha_normalized, q)  # [batch, len1, dim]

        # [batch, len1, num_perspective, dim]
        p_proj = p.unsqueeze(2).repeat(1, 1, self.num_perspective, 1) * W
        # [batch, len1, num_perspective, dim]
        p_align_proj = p_align.unsqueeze(2).repeat(1, 1, self.num_perspective, 1) * W

        rt = F.cosine_similarity(p_proj, p_align_proj, dim=-1)
        return rt

    def max_attentive_matching(self, p, q, W):
        """

        :param p: [batch, len1, dim]
        :param q: [batch, len2, dim]
        :param W: [num_perspective, dim]
        :return:
        """
        len1, len2 = p.size(1), q.size(1)
        p_expand = p.unsqueeze(2).repeat(1, 1, len2, 1)  # [batch, len1, len2, dim]
        q_expand = q.unsqueeze(1).repeat(1, len1, 1, 1)  # [batch, len1, len2, dim]

        alpha = F.cosine_similarity(p_expand, q_expand, dim=-1)  # [batch, len1, len2]
        indices = torch.max(alpha, dim=2)[1]  # [batch, len1]
        # [batch, len1, dim]
        p_align = torch.cat([torch.index_select(s, 0, i).unsqueeze(0) for i, s in enumerate(indices, q)])

        # [batch, len1, num_perspective, dim]
        p_proj = p.unsqueeze(2).repeat(1, 1, self.num_perspective, 1) * W
        # [batch, len1, num_perspective, dim]
        p_align_proj = p_align.unsqueeze(2).repeat(1, 1, self.num_perspective, 1) * W

        rt = F.cosine_similarity(p_proj, p_align_proj, dim=-1)
        return rt

    def match(self, p_for, p_back, q_for, q_back, W):
        """

        :param p_for: [batch, seq_len, dim]
        :param p_back: [batch, seq_len, dim]
        :param q_for: [batch, seq_len, dim]
        :param q_back: [batch, seq_len, dim]
        :param W: [num_perspective, dim]
        :return:
        """

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        batch = q1_word.size(0)
        w_len1, w_len2 = q1_word.size(1), q2_word.size(1)
        c_len1, c_len2 = q1_char.size(1), q2_char.size(2)
        # Word Representation Layer
        word_o1 = self.word_lstm(q1_word)[0]
        word_o2 = self.word_lstm(q2_word)[0]
        word_o1_expand = word_o1.view(batch, w_len1, 2, -1)  # [batch, seq_len, num_directions(2), hidden_dim]
        word_o1_for, word_o1_back = word_o1_expand[:, :, 0, :], word_o1_expand[:, :, 1, :]
        word_o2_expand = word_o2.view(batch, w_len2, 2, -1)
        word_o2_for, word_o2_back = word_o2_expand[:, :, 0, :], word_o2_expand[:, :, 1, :]

        char_o1 = self.char_lstm(q1_char)[0]
        char_o2 = self.char_lstm(q2_char)[0]
        char_o1_expand = char_o1.view(batch, c_len1, 2, -1)  # [batch, seq_len, num_directions(2), hidden_dim]
        char_o1_for, char_o1_back = char_o1_expand[:, :, 0, :], char_o1_expand[:, :, 1, :]
        char_o2_expand = char_o2.view(batch, c_len2, 2, -1)
        char_o2_for, char_o2_back = char_o2_expand[:, :, 0, :], char_o2_expand[:, :, 1, :]


class ESIM(BaseModel):
    def __init__(self, word_embedding, char_embedding, hidden_dim, composition_dim):
        super(ESIM, self).__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding),
            # nn.BatchNorm1d(300),
            # nn.Dropout(0.2),
            # nn.Dropout(p=0.25)
        )
        self.char_embed = nn.Sequential(
            nn.Embedding.from_pretrained(char_embedding),
            # nn.BatchNorm1d(300),
            # nn.Dropout(0.2),
            # nn.Dropout(p=0.25)
        )
        self.word_embed_bn = nn.BatchNorm1d(300)
        self.char_embed_bn = nn.BatchNorm1d(300)
        self.word_lstm1 = nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.char_lstm1 = nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.word_lstm2 = nn.LSTM(input_size=8 * hidden_dim, hidden_size=composition_dim, batch_first=True,
                                  bidirectional=True)
        self.char_lstm2 = nn.LSTM(input_size=8 * hidden_dim, hidden_size=composition_dim, batch_first=True,
                                  bidirectional=True)
        self.out = nn.Linear(16 * composition_dim, 1)
        self.initialize_weights()

    def forward(self, *input):
        q1_word, q2_word, q1_char, q2_char, feature = input
        # Input Encoding
        word_mask1, word_mask2 = q1_word.eq(1), q2_word.eq(1)
        word_e1 = self.word_embed_bn(self.word_embed(q1_word).transpose(1, 2).contiguous()).transpose(1, 2)
        word_e2 = self.word_embed_bn(self.word_embed(q2_word).transpose(1, 2).contiguous()).transpose(1, 2)
        word_o1 = self.word_lstm1(word_e1)[0]  # [batch, len1, hidden_dim]
        word_o2 = self.word_lstm1(word_e2)[0]  # [batch, len2, hidden_dim]

        char_mask1, char_mask2 = q1_char.eq(1), q2_char.eq(1)
        char_e1 = self.char_embed_bn(self.char_embed(q1_char).transpose(1, 2).contiguous()).transpose(1, 2)
        char_e2 = self.char_embed_bn(self.char_embed(q2_char).transpose(1, 2).contiguous()).transpose(1, 2)
        char_o1 = self.char_lstm1(char_e1)[0]
        char_o2 = self.char_lstm1(char_e2)[0]

        # Local Inference Modelling
        word_q1_align, word_q2_align = self.soft_attention_align(word_o1, word_o2, word_mask1, word_mask2)
        char_q1_align, char_q2_align = self.soft_attention_align(char_o1, char_o2, char_mask1, char_mask2)
        word_q1_cat = self.interact(word_o1, word_q1_align)
        word_q2_cat = self.interact(word_o2, word_q2_align)
        char_q1_cat = self.interact(char_o1, char_q1_align)
        char_q2_cat = self.interact(char_o2, char_q2_align)

        # Inference Composition
        # [batch, len1, composition_dim]
        word_c1 = self.word_lstm2(word_q1_cat)[0]
        word_c2 = self.word_lstm2(word_q2_cat)[0]
        char_c1 = self.char_lstm2(char_q1_cat)[0]
        char_c2 = self.char_lstm2(char_q2_cat)[0]

        word_v1 = self.avg_max_pool(word_c1)
        word_v2 = self.avg_max_pool(word_c2)
        char_v1 = self.avg_max_pool(char_c1)
        char_v2 = self.avg_max_pool(char_c2)

        # [batch, 4*composition_dim]
        v = torch.cat((word_v1, word_v2, char_v1, char_v2), dim=1)
        v = F.dropout(v, p=0.5, training=self.training)
        out = self.out(v)
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


class SSE(BaseModel):
    def __init__(self, word_embedding, char_embedding, dim1, dim2, dim3):
        super(SSE, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embedding)
        self.char_embed = nn.Embedding.from_pretrained(char_embedding)
        self.w_lstm1 = nn.LSTM(300, dim1, batch_first=True, bidirectional=True)
        self.w_lstm2 = nn.LSTM(300 + 2 * dim1, dim2, batch_first=True, bidirectional=True)
        self.w_lstm3 = nn.LSTM(300 + 2 * (dim1 + dim2), dim3, batch_first=True, bidirectional=True)
        self.c_lstm1 = nn.LSTM(300, dim1, batch_first=True, bidirectional=True)
        self.c_lstm2 = nn.LSTM(300 + 2 * dim1, dim2, batch_first=True, bidirectional=True)
        self.c_lstm3 = nn.LSTM(300 + 2 * (dim1 + dim2), dim3, batch_first=True, bidirectional=True)
        self.out = nn.Linear(16 * dim3, 1)

    def drop(self, x, p=0.2):
        return F.dropout(x, p=p, training=self.training)

    def forward(self, *input):
        w1, w2, c1, c2 = input
        w_e1, w_e2 = self.word_embed(w1), self.word_embed(w2)
        c_e1, c_e2 = self.char_embed(c1), self.char_embed(c2)

        # Level 1
        w1_i1, w1_i2 = F.dropout(w_e1, 0.5, self.training), F.dropout(w_e2, 0.5, self.training)
        c1_i1, c1_i2 = F.dropout(c_e1, 0.5, self.training), F.dropout(c_e2, 0.5, self.training)
        w1_o1, w1_o2 = self.w_lstm1(w1_i1)[0], self.w_lstm1(w1_i2)[0]
        c1_o1, c1_o2 = self.c_lstm1(c1_i1)[0], self.c_lstm1(c1_i2)[0]

        # Level 2
        w2_i1, w2_i2 = torch.cat((w_e1, w1_o1), dim=-1), torch.cat((w_e2, w1_o2), dim=-1)
        w2_i1, w2_i2 = F.dropout(w2_i1, 0.5, self.training), F.dropout(w2_i2, 0.5, self.training)
        c2_i1, c2_i2 = torch.cat((c_e1, c1_o1), dim=-1), torch.cat((c_e2, c1_o2), dim=-1)
        c2_i1, c2_i2 = F.dropout(c2_i1, 0.5, self.training), F.dropout(c2_i2, 0.5, self.training)
        w2_o1, w2_o2 = self.w_lstm2(w2_i1)[0], self.w_lstm2(w2_i2)[0]
        c2_o1, c2_o2 = self.c_lstm2(c2_i1)[0], self.c_lstm2(c2_i2)[0]

        # Level 3
        w3_i1, w3_i2 = torch.cat((w_e1, w1_o1, w2_o1), dim=-1), torch.cat((w_e2, w1_o2, w2_o2), dim=-1)
        w3_i1, w3_i2 = F.dropout(w3_i1, 0.5, self.training), F.dropout(w3_i2, 0.5, self.training)
        c3_i1, c3_i2 = torch.cat((c_e1, c1_o1, c2_o1), dim=-1), torch.cat((c_e2, c1_o2, c2_o2), dim=-1)
        c3_i1, c3_i2 = F.dropout(c3_i1, 0.5, self.training), F.dropout(c3_i2, 0.5, self.training)
        w3_o1, w3_o2 = self.w_lstm3(w3_i1)[0], self.w_lstm3(w3_i2)[0]
        c3_o1, c3_o2 = self.c_lstm3(c3_i1)[0], self.c_lstm3(c3_i2)[0]

        w1_v, w2_v = w3_o1.max(dim=1)[0], w3_o2.max(dim=1)[0]
        c1_v, c2_v = c3_o1.max(dim=1)[0], c3_o2.max(dim=1)[0]

        w_v, c_v = self.interact(w1_v, w2_v), self.interact(c1_v, c2_v)
        v = F.dropout(torch.cat((w_v, c_v), dim=-1), 0.5, self.training)
        out = self.out(v)
        return F.sigmoid(out)


class ESIM_word(BaseModel):
    def __init__(self, word_embedding, encode_dim, composition_dim):
        super(ESIM_word, self).__init__()
        self.embed = nn.Sequential(
            nn.Embedding.from_pretrained(word_embedding),
            nn.Dropout(p=0.2)
        )
        self.lstm1 = nn.LSTM(input_size=300, hidden_size=encode_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=8 * encode_dim, hidden_size=composition_dim, batch_first=True,
                             bidirectional=True)
        self.out = nn.Linear(16 * composition_dim, 1)

    def forward(self, *input):
        q1, q2 = input[0], input[1]
        mask1, mask2 = q1.eq(0), q2.eq(0)

        # Input Encoding
        x1 = self.embed(q1)
        x2 = self.embed(q2)
        o1 = F.dropout(self.lstm1(x1)[0], p=0.25, training=self.training)
        o2 = F.dropout(self.lstm1(x2)[0], p=0.25, training=self.training)

        # Local Inference Modelling
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)  # [batch, len,2*encode_dim]
        q1_cat = self.interact(o1, q1_align)  # [batch, len1, 8*encode_dim]
        q2_cat = self.interact(o2, q2_align)  # [batch, len2, 8*encode_dim]

        # Inference Composition
        c1 = self.lstm2(q1_cat)[0]  # [batch, len1, 2*composition_dim]
        c2 = self.lstm2(q2_cat)[0]
        v1 = self.avg_max_pool(c1)  # [batch, 4*composition_dim]
        v2 = self.avg_max_pool(c2)
        v = F.dropout(self.interact(v1, v2, dim=-1), p=0.5, training=self.training)  # [batch, 8*composition_dim]
        out = self.out(v)
        return F.sigmoid(out)


class ABCNN(nn.Module):
    pass


if __name__ == '__main__':
    word_vector_path = '../data/word_vectors.txt'
    char_vector_path = '../data/char_vectors.txt'

    # CV: 0.8572 LB: 0.85251
    # model_fn = lambda word_embedding, char_embedding: InferSent(word_embedding, char_embedding, 64, 256, 128)
    # for seed in range(11, 31):
    #     fit(model_fn, 'InferSent', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8642 LB: ?
    # model_fn = lambda word_embedding, char_embedding: InferSent_Feat(word_embedding, char_embedding, 64, 256, 128)
    # for seed in range(1, 11):
    #     fit(model_fn, 'InferSent_Feat', word_vector_path, char_vector_path, 10, seed=seed)

    # CV: 0.8531 LB: 0.84024
    # model_fn = lambda word_embedding, char_embedding: SiameseTextCNN(word_embedding, char_embedding, 64, 128)
    # for seed in range(2, 11):
    #     fit(model_fn, 'SiameseTextCNN_1124', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8570 LB: ?
    # model_fn = lambda word_embedding, char_embedding: SiameseTextCNN(word_embedding, char_embedding, 64, 128,
    #                                                                  use_feature=True)
    # for seed in range(1, 11):
    #     fit(model_fn, 'SiameseTextCNN_Feat_1124', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64, 128,
                                                                            True)
    for seed in range(8, 11):
        fit(model_fn, 'DecomposableAttention_Feat_1125', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # CV: 0.8529 LB: 0.83794
    model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64, 128)
    for seed in range(1, 11):
        fit(model_fn, 'DecomposableAttention_1125', word_vector_path, char_vector_path, num_folds=10, seed=seed)

    # model_fn = lambda word_embedding, char_embedding: InferSent_Feat_v1(word_embedding, char_embedding, 64, 256, 128)
    # for seed in range(1, 11):
    #     fit(model_fn, 'InferSent_Feat_v1', num_folds=10, seed=seed)
    # pass
    # model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64, 128)
    # fit(model_fn, 'DecomposableAttention', granularity='concat', seed=6)

    # model_fn = lambda word_embedding, char_embedding: DecomposableAttention(word_embedding, char_embedding, 50, 64, 128)
    # for seed in range(2, 6):
    #     fit(model_fn, 'DecomposableAttention', granularity='concat', seed=seed)

    # model_fn = lambda word_embedding, char_embedding: ESIM(word_embedding, char_embedding, 64, 64)
    # fit(model_fn, 'ESIM_test', word_vector_path, char_vector_path, 10)

    # model_fn = lambda word_embedding, char_embedding: LexicalDecomposition(word_embedding, char_embedding, 64)
    # fit(model_fn, 'LexicalDecomposition', word_vector_path, char_vector_path, 10)

    # model_fn = lambda word_embedding, char_embedding: ESIM(word_embedding, char_embedding, 64, 64)
    # fit(model_fn, 'ESIM', granularity='concat', seed=2, lr=5e-4)

    # model_fn = lambda word_embedding, char_embedding: InferSent(word_embedding, char_embedding, 64, 256, 128)
    # for seed in range(2, 16):
    #     fit(model_fn, 'InferSent_tuneEmbed', word_vector_path, char_vector_path, 10, seed=seed)

    pass
