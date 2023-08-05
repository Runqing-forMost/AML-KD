from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class,log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all':
                self.train_imgs = train_imgs          
                    
    def __getitem__(self, index):
        if self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+ '/google_resized_256/'+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target, index
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    



class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir, log=''):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", num_class=self.num_class)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)        
            
            unlabeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",num_class=self.num_class,pred=pred,log=self.log)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            
        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                   
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs) 
                 
        elif self.mode == "labeled":   
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path, index        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target, index
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
class clothing_dataloader():  

    def __init__(self, root, batch_size, num_batches, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])        
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=64,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True)             
            return test_loader             
        elif mode=='val':
            val_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=32,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True)             
            return val_loader     