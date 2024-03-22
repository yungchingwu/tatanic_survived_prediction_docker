import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class my_one_hot_encoder:
    def __init__(self):
        self.one_hot_encoder = None
        self.one_hot_encoder_table = None

    def create_one_hot_encoder(self, df: pd.DataFrame, encode_label: list) -> OneHotEncoder:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(df[encode_label])

        self.one_hot_encoder = encoder
        self.one_hot_encoder_table = encode_label

        return encoder

    def apply_one_hot_encoder(self, df: pd.DataFrame, encode_label: list, encoder: OneHotEncoder) -> pd.DataFrame:
        df_encoded = encoder.transform(df[encode_label]).toarray()
        feature_names = encoder.get_feature_names_out(encode_label)
        df_encoded = pd.concat([df, pd.DataFrame(df_encoded, columns=feature_names)], axis=1)
        df_encoded = df_encoded.drop(encode_label, axis=1)
        return df_encoded

    def apply_my_one_hot_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = self.one_hot_encoder.transform(df[self.one_hot_encoder_table]).toarray()
        feature_names = self.one_hot_encoder.get_feature_names_out(self.one_hot_encoder_table)
        df_encoded = pd.concat([df, pd.DataFrame(df_encoded, columns=feature_names)], axis=1)
        df_encoded = df_encoded.drop(self.one_hot_encoder_table, axis=1)
        return df_encoded
