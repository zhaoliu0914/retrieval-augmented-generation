import csv
import json

if __name__ == '__main__':
    # with open("data/202401.json", "r") as file:
    #     data = json.load(file)
    #     print(data[0])

    with open('data/202411.csv', mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        header = next(csv_reader)
        print(f"Header: {header}")

        for row in csv_reader:
            print(row)