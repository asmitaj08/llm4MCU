if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="evaluated_output.csv")
    parser.add_argument("--gt_column", type=str, default="groundtruth")
    # parser.add_argument("--answer_column", type=str, default="ans1") #multiple ans coloumn
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    result_df = evaluate_dataframe(df, pred_col=args.answer_column, gt_col=args.gt_column, batch_size=args.batch_size, workers=args.workers)
    result_df.to_csv(args.output_csv, index=False)
    print(f"Saved evaluated CSV to {args.output_csv}")
