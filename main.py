import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args, resume=False):

    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    if resume:
        checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, scheduler = build_optimizer(args, model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scheduler.last_epoch = checkpoint['epoch']
        best_score = checkpoint['mean_f1']
        init = checkpoint['epoch'] + 1
        step = checkpoint['step']
        del checkpoint
    else:
        optimizer, scheduler = build_optimizer(args, model)
        best_score = args.best_score
        init = 0
        step = 0

    if args.device == 'cuda' and not resume:
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    for epoch in range(init, args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            step += 1

            # if step % args.gradient_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # if step % args.print_steps == 0:
            time_per_step = (time.time() - start_time) / max(1, step)
            remaining_time = time_per_step * (num_total_steps - step)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            # state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save(
                {
                    'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if args.device == 'cuda' else model.state_dict(), 
                    'mean_f1': mean_f1,
                    'step':step,
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict()
                },
                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin'
                    )


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    # train_and_validate(args, False)
    train_and_validate(args, True)


if __name__ == '__main__':
    main()
