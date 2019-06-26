import argparse
import os
import numpy as np
import math
import random
import pdb

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from dataset import CartoonDataset

from torchvision import datasets
from torch.autograd import Variable

from torch.nn.utils import spectral_norm
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch






parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--test", action='store_true', help="used to check if it is testing FID")
parser.add_argument("--best_ckpt_path", type=str, help="the best checkpoint's path")
parser.add_argument("--testing_file", type=str, help="the testing file")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00008, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--data_dir', type=str, default='selected_cartoonset100k/images_data/', help='image file of dataset')
parser.add_argument('--label_txt', type=str, default='selected_cartoonset100k/cartoon_attr.txt', help='Txt file of dataset')
parser.add_argument('--output_dir', type=str, default='Result/', help='output directory of results')
parser.add_argument('--model_dir', type=str, default='checkpoint_AC_v2/', help='output directory of generator')

opt = parser.parse_args()
os.makedirs(opt.output_dir, exist_ok=True)
print(opt)

cuda = True if torch.cuda.is_available() else False
# device = torch.device("cuda" if cuda else "cpu")
device = torch.device("cuda")
# save information of all images
train_filename = []
train_image_path = {}
train_image_label = {}

test_image_label = []

all_lines = []
with open(opt.testing_file) as f:
    for line in f:
        all_lines.append(line)
if opt.test:
    for i,line in enumerate(all_lines):
        if i == 0 or i == 1:
            continue
        else:
            line = line.split()
            line = list(map(float, line))
            test_image_label.append(line)
else:
    for i,line in enumerate(all_lines[:146]):
        if i == 0 or i == 1:
            continue
        else:
            line = line.split()
            line = list(map(float, line))
            test_image_label.append(line)
test_image_label = torch.from_numpy(np.array(test_image_label)).float().to(device)

def loop_all_file(path):
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            # f = complete path to the image
            train_filename.append(filename)
            f = os.path.join(path, filename)
            train_image_path[filename] = f
            # construct train_image_label to save features from label.txt at the same time
            train_image_label[filename] = []
        elif filename.endswith(".PNG"):
            train_filename.append(filename)
            f = os.path.join(path, filename)
            # f = complete path to the image
            train_image_path[filename] = f
            # construct train_image_label to save features from label.txt at the same time
            train_image_label[filename] = []
        else:
            pass


def preprocess(data_dir, label_txt):
    loop_all_file(data_dir)
    with open(label_txt, 'r') as f:
        lines = f.readlines()
        data_num = int(lines[0]) # total number of 
        for i in range(data_num):
            line = lines[2+i]
            data = line.split()
            f_name = data[0]
            train_image_label[f_name] = data[1:]

# execute the preprocess function to extract the info in label.txt
if not opt.test:
    preprocess(opt.data_dir, opt.label_txt)
    for i in train_image_label:
        tmp = train_image_label[i]
        for j in range(len(tmp)):
            tmp[j] = float(tmp[j])
        train_image_label[i] = tmp
    # construct the dataset
    t = transforms.Compose([transforms.ToTensor()])
    cartoon = CartoonDataset(train_filename, path_dict=train_image_path, label_dict=train_image_label, transforms=t)

    dataloader = torch.utils.data.DataLoader(cartoon, batch_size=opt.batch_size, shuffle=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, dim=128):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim + c_dim, dim * 4, 4, 1, 0, 0),  # (N, dim * 4, 4, 4)
            dconv_bn_relu(dim * 4, dim * 2, 4, 4, 0, 0),  # (N, dim * 2, 16, 16)
            dconv_bn_relu(dim * 2, dim, 4, 4, 0, 0),  # (N, dim, 64, 64)
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1), nn.Tanh()  # (N, 3, 128, 128)
        )
        

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        x = torch.cat([z, c.float()], 1)
        x = self.ls(x.view(x.size(0), x.size(1), 1, 1))
        # print(x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        # one hot for concat : opt.latent_dim -> opt.n_classes
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2+15, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes))

    def forward(self, img, label):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        
        v_out = torch.cat((out, label), dim=1)
        validity = self.adv_layer(v_out)
        label = self.aux_layer(out)

        return validity,label


# Loss functions

auxiliary_loss = torch.nn.CrossEntropyLoss()

def adversarial_loss_d(r_logit, f_logit):
    r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
    f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
    return r_loss, f_loss

def adversarial_loss_g(f_logit):
    f_loss = - f_logit.mean()
    return f_loss

# Initialize generator and discriminator
generator = Generator(opt.latent_dim,opt.n_classes)
discriminator = Discriminator()


# if cuda:
generator.cuda()
discriminator.cuda()

auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    z = Variable(FloatTensor(np.random.normal(0, 1, (144, opt.latent_dim))))
    
    gen_imgs = generator(z, test_image_label)
    path = os.path.join(opt.output_dir, '{}.png'.format(batches_done))
    # path = opt.output_dir + "images/%d.png" % batches_done
    gen_imgs.data = (gen_imgs.data+1)/2
    save_image(gen_imgs.data, path, nrow=n_row)


def save(path):
    torch.save({
        'model': generator.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, path)

def load(path):
        checkpoint = torch.load(path)
        generator.load_state_dict(checkpoint['model'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        

# ----------
#  Training
# ----------
if opt.test:
    # all_lines2 = []
    # FID = []
    # with open('sample_fid_testing_labels.txt') as f1:
    #     for line in f1:
    #         all_lines2.append(line)
    # for i,line in enumerate(all_lines2[2:]):
    #     line = line.split()
    #     line = list(map(float, line))
    #     FID.append(line)
    # FID = torch.from_numpy(np.array(FID)).float().to(device)
    #FID (5000,15)
    load(opt.best_ckpt_path)
    cnt = 0
    # for i in range(0,FID.size(0),32):
    #     if i == 4992:
    #         z2 = Variable(FloatTensor(np.random.normal(0, 1, (8, opt.latent_dim))))
    #     else:
    #         z2 = Variable(FloatTensor(np.random.normal(0, 1, (32, opt.latent_dim))))
    #     gen_imgs = generator(z2, FID[i:i+32,:])
    #     gen_imgs.data = (gen_imgs.data+1)/2
    #     for gen_img in gen_imgs:
    #         path = os.path.join(opt.output_dir, '{}.png'.format(cnt))
    #         cnt += 1
    #         save_image(gen_img.data, path)
    for i in range(len(test_image_label)):
        z2 = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        gen_imgs = generator(z2, test_image_label[i:i+1,:])
        gen_imgs.data = (gen_imgs.data+1)/2
        path = os.path.join(opt.output_dir, '{}.png'.format(cnt))
        cnt += 1
        save_image(gen_imgs.data, path)

        # save_image(gen_imgs.data, path, nrow=n_row)

else:
    for epoch in range(opt.n_epochs):
        # run batches
        gloss = 0
        dloss = 0
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="GOGOGOGO")

        for i, (imgs, labels) in trange:
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # get the corresponding index to turn embedding in the generator
            # # fake label for generator
            # gen_hair_idx = torch.tensor(np.random.randint(0, 6, batch_size), device=device, dtype=torch.long)
            # gen_eye_idx = torch.tensor(np.random.randint(0, 4, batch_size), device=device, dtype=torch.long)
            # gen_face_idx = torch.tensor(np.random.randint(0, 3, batch_size), device=device, dtype=torch.long)
            # gen_glasses_idx = torch.tensor(np.random.randint(0, 2, batch_size), device=device, dtype=torch.long)

            # gen_labels = []
            # for j in range(batch_size): 
            #     a1 = [0,0,0,0,0,1]
            #     a2 = [0,0,0,1]
            #     a3 = [0,0,1]
            #     a4 = [0,1]
            #     random.shuffle(a1)
            #     random.shuffle(a2)
            #     random.shuffle(a3)
            #     random.shuffle(a4)
            #     gen_labels.append(a1+a2+a3+a4)
            # gen_labels = torch.from_numpy(np.array(gen_labels)).float().to(device)
            # real label
            label_hair_idx = torch.argmax(labels[:,0:6], dim=1)
            label_eye_idx = torch.argmax(labels[:,6:10], dim=1)
            label_face_idx = torch.argmax(labels[:,10:13], dim=1)
            label_glasses_idx = torch.argmax(labels[:,13:15], dim=1)

            # gen_hair_idx = torch.argmax(gen_labels[:,0:6], dim=1)
            # gen_eye_idx = torch.argmax(gen_labels[:,6:10], dim=1)
            # gen_face_idx = torch.argmax(gen_labels[:,10:13], dim=1)
            # gen_glasses_idx = torch.argmax(gen_labels[:,13:15], dim=1)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
    #         z = Variable(FloatTensor(np.random.binomial(144, 0.01, (batch_size, opt.latent_dim))))

            
            # Generate a batch of images
            gen_imgs = generator(z, labels)
            # gen_imgs, label_emb = generator(z, label_hair_idx, label_eye_idx, label_face_idx, label_glasses_idx)

            # Loss measures generator's ability to fool the discriminator

            # use one hot directly
            # validity, pred_hair, pred_eye, pred_face, pred_glasses = discriminator(gen_imgs, labels)
            validity, pred_label = discriminator(gen_imgs, labels)
    #         validity, pred_hair, pred_eye, pred_face, pred_glasses = discriminator(gen_imgs, label_emb)


    #         auxi_loss = (auxiliary_loss(pred_hair, gen_hair_idx) + auxiliary_loss(pred_eye, gen_eye_idx) + auxiliary_loss(pred_face, gen_face_idx) + auxiliary_loss(pred_glasses, gen_glasses_idx)) / 4
    #         auxi_loss = (auxiliary_loss(pred_hair, label_hair_idx) + auxiliary_loss(pred_eye, label_eye_idx) + auxiliary_loss(pred_face, label_face_idx) + auxiliary_loss(pred_glasses, label_glasses_idx)) / 4
    #         g_loss = (adversarial_loss(validity, valid) + auxi_loss) / 2
    #         g_loss = adversarial_loss(validity, valid)

    #         g_loss.backward()
    #         optimizer_G.step()
    #     if i % 10 ==0:
    #         optimizer_G.zero_grad()
            auxi_loss = (auxiliary_loss(pred_label[:,:6], label_hair_idx) \
                + auxiliary_loss(pred_label[:,6:10], label_eye_idx) \
                + auxiliary_loss(pred_label[:,10:13], label_face_idx) \
                + auxiliary_loss(pred_label[:,13:15], label_glasses_idx))
    #         g_iloss = adversarial_loss(validity, valid) 
            g_loss = (adversarial_loss_g(validity) + auxi_loss)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            # pdb.set_trace()
            imgs = imgs.to(device)
            real_pred, real_label = discriminator(imgs, labels.detach())
            real_auxi_loss = (auxiliary_loss(real_label[:,:6], label_hair_idx) \
                + auxiliary_loss(real_label[:,6:10], label_eye_idx) \
                + auxiliary_loss(real_label[:,10:13], label_face_idx) \
                + auxiliary_loss(real_label[:,13:15], label_glasses_idx))
    #         auxi_loss =  auxiliary_loss(real_hair, label_hair_idx) + auxiliary_loss(real_eye, label_eye_idx) + auxiliary_loss(real_face, label_face_idx) + auxiliary_loss(real_glasses, label_glasses_idx)
            
    #         d_real_loss = adversarial_loss(real_pred, valid)
        
            # Loss for fake images
            fake_pred, fake_label = discriminator(gen_imgs.detach(), labels.detach()) # no back-propagation so use detach
    #         auxi_loss = (auxiliary_loss(fake_hair, gen_hair_idx) + auxiliary_loss(fake_eye, gen_eye_idx) + auxiliary_loss(fake_face, gen_face_idx) + auxiliary_loss(fake_glasses, gen_glasses_idx)) / 4
            fake_auxi_loss = (auxiliary_loss(fake_label[:,:6], label_hair_idx) \
                + auxiliary_loss(fake_label[:,6:10], label_eye_idx) \
                + auxiliary_loss(fake_label[:,10:13], label_face_idx) \
                + auxiliary_loss(fake_label[:,13:15], label_glasses_idx))
    #         d_fake_loss = adversarial_loss(fake_pred, fake)
            # Total discriminator loss
    #         d_loss = (d_real_loss + d_fake_loss) / 2
    #         d_fake_loss = adversarial_loss(fake_pred, fake)
            d_fake_loss,d_real_loss = adversarial_loss_d(real_pred, fake_pred) 
            d_loss = d_real_loss + d_fake_loss + real_auxi_loss + fake_auxi_loss
            
            # Here is the new way to calculate loss 
    #         d_loss = d_fake_loss + auxi_loss + d_real_loss


            d_loss.backward()
            optimizer_D.step()
            
            gloss += (g_loss.item() / 5)
            dloss += (d_real_loss.item() / 5) + (d_fake_loss.item() / 5)

            trange.set_postfix({"epoch":"{}".format(epoch),"g_loss":"{0:.5f}".format(gloss / (i + 1)), "d_loss":"{0:.5f}".format(dloss / (i + 1))})

            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=12, batches_done=epoch)
                if epoch > 800:
                    model_path = os.path.join(opt.model_dir, 'model.pkl.{}'.format(epoch))
                    save(model_path)





