import tensorflow as tf
import pandas as pd


def load_parquet_as_tf_dataset(parquet_path, batch_size=32, shuffle=True):
    df = pd.read_parquet(parquet_path)

    def generator():
        for _, row in df.iterrows():
            x = row['feature']
            y = row['label']
            yield x, y

    output_types = (tf.float32, tf.int64)  # 修改成你数据的类型
    dataset = tf.data.Dataset.from_generator(generator, output_types=output_types)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    return dataset.batch(batch_size)
