import torch
import wget

def preprocess(model, name='dino', embed_dim=384, patch_size=16):
    new_model = {}
    for k in model.keys():
        if 'patch_embed.proj.weight' in k:
            x = torch.zeros(embed_dim, 4, patch_size, patch_size)
            x[:, :3] = model[k]
            new_model['backbone.'+k] = x
        else:
            new_model['backbone.'+k] = model[k]
    if embed_dim==384:
        size='s'
    else:
        size='b'
    torch.save(new_model, name+'_vit_'+ size + '_fna.pth')


if __name__ == "__main__":

    # wget.download('https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth')
    # dino_model = torch.load('dino_deitsmall16_pretrain.pth')
    # preprocess(dino_model, 'dino', 384)

    # wget.download('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth')
    # mae_model = torch.load('mae_pretrain_vit_base.pth')['model']
    # preprocess(mae_model, 'mae', 768)

    # wget.download('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth')
    dinov2_model = torch.load('dinov2_vitb14_pretrain.pth')
    import pdb; pdb.set_trace()
    preprocess(dinov2_model, 'dinov2', 768, patch_size=14)
