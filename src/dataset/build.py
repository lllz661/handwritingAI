import argparse
from prepare import prepare_dataset
from split import split_dataset

def main():
    parser = argparse.ArgumentParser(description="Dataset pipeline")
    parser.add_argument("--source", default="dataset/ground_truth/images", help="Путь к исходным изображениям")
    parser.add_argument("--labels", default="dataset/ground_truth/labels", help="Путь к аннотациям")
    parser.add_argument("--output", default="dataset/prepared", help="Куда сохранить результат")
    parser.add_argument("--test-size", type=float, default=0.1, help="Размер тестовой выборки")
    parser.add_argument("--val-size", type=float, default=0.15, help="Размер валидации")
    parser.add_argument("--img-size", type=int, default=640, help="Размер изображений")
    args = parser.parse_args()

    print("🔹 Шаг 1. Препроцессинг...")
    prepare_dataset(args.source, args.labels, args.output, img_size=args.img_size)

    print("🔹 Шаг 2. Разделение...")
    split_dataset(args.output, val_size=args.val_size, test_size=args.test_size)

    print("✅ Датасет готов!")

if __name__ == "__main__":
    main()
