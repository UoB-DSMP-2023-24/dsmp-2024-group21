import pandas as pd
import multiprocessing

def worker(task):
    # 这里是处理每个任务的逻辑
    print(f"Processing {task}")
    # 模拟数据处理
    # 注意：实际应用中，你需要根据具体任务替换这部分代码
    return f"Processed {task}"

def init(l):
    global lock
    lock = l

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {cpu_count}")

    # 创建一个锁
    lock = multiprocessing.Lock()

    # 创建一个进程池
    with multiprocessing.Pool(initializer=init, initargs=(lock,), processes=cpu_count) as pool:
        tasks = [f"Task {i}" for i in range(10)]  # 示例任务列表
        results = pool.map(worker, tasks)
        for result in results:
            # 使用锁来同步数据写入操作
            with lock:
                print(result)  # 这里模拟数据写入操作


def load_LOBs_data(directory_path):
    # Use regex to select filename pattern and identify the date
    file_pattern = re.compile(r'^UoB_Set01_(\d{4}-\d{2}-\d{2})LOBs\.txt$')
    for filename in os.listdir(directory_path):
        match = file_pattern.match(filename)
        if match:
            print(f"Processing file: {filename}")
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r') as file:
                data_rows = []  # List to collect data rows
                for line in file:
                    # Directly evaluate the line as a Python list
                    data_list = ast.literal_eval(preprocess_line(line.strip()))
                    # Extract components directly from the list
                    timestamp, exchange, orders = data_list
                    bids, asks = parse_orders(str(orders))
                    # Prepare the row and add it to the list
                    # encode bid to 0 and ask to 1
                    for price, quantity in bids:
                        data_rows.append({'date': date_obj, 'timestamp': timestamp,
                                         'type': 'bid', 'price': price, 'quantity': quantity})
                    for price, quantity in asks:
                        data_rows.append({'date': date_obj, 'timestamp': timestamp,
                                         'type': 'ask', 'price': price, 'quantity': quantity})
                print(f"Data rows length: {len(data_rows)}") 
                df = pd.DataFrame(data_rows, columns=[
                        'date', 'timestamp', 'type', 'price', 'quantity'])
                # Convert 'type' to 0 for bids and 1 for asks
                df['type'] = df['type'].map({'bid': 0, 'ask': 1}).astype(np.int8)
                # Convert 'price' and 'quantity' to suitable numeric types
                df['price'] = df['price'].astype(np.float32)
                df['quantity'] = df['quantity'].astype(np.int32)
                print(df.describe())
                append_hdf5(df,hdf5_path=LOB_hdf5_path,key='lob')
    return 
