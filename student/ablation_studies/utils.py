def log_train_test_split(csv_writer, iteration, train_loss, test_loss):
    csv_writer.writerow([iteration, train_loss, test_loss])
