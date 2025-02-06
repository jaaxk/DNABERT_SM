import torch 

def get_optimizer(model, args):
    base_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.Adam([
            {'params':base_model.dnabert2.parameters()}, 
            {'params':base_model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
            {'params':base_model.attention.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    return optimizer 
    

