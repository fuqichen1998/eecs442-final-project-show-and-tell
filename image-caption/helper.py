import torch

def save_checkpoint(encoder, decoder, encoder_opt, decoder_opt, dataset, epoch_num, bleu, is_highest_score):
    state = {
        'encoder': encoder,
        'decoder': decoder,
        'encoder_opt': encoder_opt,
        'decoder_opt': decoder_opt,
        'epoch': epoch_num,
        'bleu-score': bleu
    }

    filename = 'checkpoint_' + dataset + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_highest_score:
        torch.save(state, 'best_' + filename)
