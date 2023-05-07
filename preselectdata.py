import pandas as pd
import shutil  # 用于文件复制
import os

def select_data():
    source_folder = 'D:\\asassnvarlc_vband_complete\\vardb_files'
    dest_folder = 'D:\\asas_select_data'
    filename = 'data/Stars_Catalog.csv'
    # source_folder = 'G:\\A1\\asassnvarlc_vband_complete-001\\vardb_files'
    # dest_folder = 'G:\\A1\\asas_select_data_other'

    # 读取csv文件为DataFrame
    df = pd.read_csv(filename)

    # 处理asas_name列数据
    df['asassn_name'] = df['asassn_name'].str.replace(' ', '')  # 删除空格

    # 处理source_id项
    df = df[df['source_id'].str.startswith('AP')]

    exist = tuple(df['asassn_name'])

    # 将筛选结果复制到另一个文件夹中
    cnt = 0
    for filename in os.listdir(source_folder):
        print(filename)
        cnt += 1
        if cnt%1000 == 0:
            print(cnt/1000)
        if filename.endswith('.dat'):
            file_path = os.path.join(source_folder, filename)
            # 根据asas_name列筛选文件
            try:
                fn = filename[:filename.rfind('.')]
                exist.index(fn)
            except ValueError:
                continue
            # 复制文件到指定目录
            shutil.copy(file_path, os.path.join(dest_folder, filename))
    print("over!"+cnt)

if __name__ == "__main__":
    print("start")
    select_data()
