from .clip_paper_dataset import get_shadow_cc3m


def get_shadow_dataset(args):
    if args.shadow_dataset != 'cc3m':
        raise ValueError('Only the CC3M shadow dataset is supported.')
    return get_shadow_cc3m(args)
