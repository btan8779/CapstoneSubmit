import os
from Config_doc.logger import get_logger
from model.Model import get_model,define_G

import shutil
import torch
import importlib
from torch import optim,nn
from torch.nn import BCELoss, L1Loss,MSELoss

from data.dataloader import get_train_loaders
from skimage.metrics import normalized_root_mse
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MeanSquaredError
import copy
from model.tensorboard import DefaultTensorboardFormatter
from torch.utils.tensorboard import SummaryWriter

import tqdm
import itertools

logger = get_logger('Trainer')

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def NRMSE(input, target):
        input = input.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return normalized_root_mse(target, input, normalization='min-max')*100.0


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.5, 0.999)))
    # print(model)

    if isinstance(model, dict):
        # print(itertools.chain(model[mo].parameters() for mo in model.keys()))
        # all_parameters = list(itertools.chain(model[mo].parameters() for mo in model.keys()))
        optimizer = optim.Adam(itertools.chain(*[list(model[mo].parameters()) for mo in model.keys()]), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)

def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)



def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    # modified state name
    # model_state_dict = state[model_key]
    # modified_model_state_dict = {}
    # for key, value in model_state_dict.items():
    #     new_key = 'pre_net.'+key
    #     modified_model_state_dict[new_key]=value

    model.load_state_dict(state[model_key])
    # model.load_state_dict(modified_model_state_dict,strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

def create_ARGAN_3d_trainer(config):
    # Get the model we need
    generate_model = define_G(**config['generator'])
    # generate_model = get_model(config['generator'])
    refine_model = get_model(config['refine_model'])
    discriminate_model = get_model(config['discriminator'])
    # Get the device
    device = torch.device(config['device'])
    logger.info(f"Sending the model to '{config['device']}'")
    # device_ids = [0,1]
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        generate_model = nn.DataParallel(generate_model)#, device_ids=device_ids)
        refine_model = nn.DataParallel(refine_model)#, device_ids=device_ids)
        discriminate_model = nn.DataParallel(discriminate_model)#, device_ids=device_ids)

        # model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    # Put the model on the device
    generate_model = generate_model.to(device)
    refine_model = refine_model.to(device)
    discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()
    criterion_content = L1Loss()
    # Create evaluation metrics
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)

    optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']
    # lambda_content_prenet = trainer_config['lambda_content_prenet']
    # lambda_content_arnet = trainer_config['lambda_content_arnet']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return ARGANTrainer(
                         generate_model=generate_model,
                         refine_model=refine_model,
                         dis_model=discriminate_model,
                         optimizer_pre=optimizer_pre,
                        #  lr_scheduler_pre=lr_scheduler_pre,
                         optimizer_refine=optimizer_refine,
                        #  lr_scheduler_refine=lr_scheduler_refine,
                         optimizer_disc=optimizer_disc,
                         adv_loss = criterion_adv,
                         content_loss = criterion_content,
                         psnr = psnr,
                         mse = mse,
                         ssim = ssim,
                         tensorboard_formatter=tensorboard_formatter,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         lr = lr,
                         **trainer_config)



class ARGANTrainer:
    def __init__(self, generate_model, refine_model,dis_model, optimizer_pre,optimizer_disc,optimizer_refine,
                 lr, adv_loss, content_loss,psnr,
                 mse,ssim,device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.generate = generate_model
        self.refine = refine_model
        self.discriminate = dis_model
        self.optimizer_pre = optimizer_pre
        # self.scheduler_pre = lr_scheduler_pre
        self.optimizer_refine = optimizer_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizer_disc = optimizer_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generate, self.optimizer_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generate, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]
    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0,self.max_num_epochs)
        for epoch in tqdm.tqdm(
                    enumerate(epoch_list), total=self.max_num_epochs,
                    desc='Train epoch==%d' % self.num_epochs, ncols=80,
                    leave=False):

            lr = self.check_lr(epoch[1],decay_epoch=30)

            for param_group in self.optimizer_pre.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_refine.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_disc.param_groups:
                param_group['lr'] = lr


            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self,epoch,decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse= RunningAverage()
        # train_nrmse= RunningAverage()


        self.generate.train()
        self.refine.train()
        self.discriminate.train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations,self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images , spet_images, weight = self._split_training_batch(t)

            # PreNet: Generate preliminary predictions
            preliminary_predictions = self.generate(lpet_images)
            # AR-Net: Generate rectified parameters and estimated residual
            estimated_residual = self.refine(preliminary_predictions)
            # estimated_residual = preliminary_predictions * rectified_parameters

            # Combine estimated residual with preliminary predictions
            rectified_spet_like_images = preliminary_predictions+estimated_residual

            # print(rectified_spet_like_images.shape)

            # AdvNet: Train AdvNet with real and fake image pairs
            real_images = spet_images
            fake_images = rectified_spet_like_images
            # print(real_images,fake_images)
            # logger.info(f'image size:{lpet_images.shape,fake_images.shape,real_images.shape}')
            # print(real_images.shape)

            advnet_real_output = self.discriminate(torch.cat((lpet_images,real_images),dim=1))
            advnet_fake_output = self.discriminate(torch.cat((lpet_images,fake_images),dim=1)) # (fake_images.detach())

            adv_real_loss = torch.mean((advnet_real_output-1)**2)
            adv_fake_loss = torch.mean(advnet_fake_output**2)

            adv_loss = adv_real_loss+adv_fake_loss
            # print(f'adv_loss: {adv_loss}')

            # AR-Net: Train AR-Net with content and adversarial losses
            content_loss_prenet = self.lambda_content_prenet * self.content_loss(preliminary_predictions, spet_images)
            # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
            # print(f'content_loss_prenet{content_loss_prenet}')

            real_residual = spet_images - preliminary_predictions
            content_loss_arnet = self.lambda_content_arnet * self.content_loss(real_residual, estimated_residual)
            # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )
            # print(f'content_loss_arnet{content_loss_arnet}')


            arnet_loss =adv_loss + content_loss_prenet + content_loss_arnet
            train_loss.update(arnet_loss.item(),self._batch_size(lpet_images))
            # print(f'arnet_loss{arnet_loss}')
            psnr = self.psnr(fake_images,real_images)
#             train_psnr.update(psnr,self._batch_size(lpet_images))
            mse = self.mse(fake_images,real_images)
#             train_mse.update(mse,self._batch_size(lpet_images))
            nrmse = NRMSE(fake_images,real_images)
#             train_nrmse.update(nrmse,self._batch_size(lpet_images))
            ssim = self.ssim(fake_images,real_images)
#             train_ssim.update(ssim,self._batch_size(lpet_images))

            self.optimizer_pre.zero_grad()
            self.optimizer_refine.zero_grad()
            self.optimizer_disc.zero_grad()
            # print('finish zero grad')
            arnet_loss.backward()
            # print('finish backward')
            self.optimizer_pre.step()
            self.optimizer_refine.step()
            self.optimizer_disc.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the models in eval mode
                self.generate.eval()
                self.refine.eval()
                self.discriminate.eval()
                # evaluate on validation set
                val_result = self.validate()
                # set the model back to training mode
                self.generate.train()
                self.refine.train()
                self.discriminate.train()

                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss'])
                # is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')

            if self.num_iterations % self.log_after_iters == 0:
                # train_loss.update(arnet_loss.item(), self._batch_size(lpet_images))
                # train_psnr.update(psnr, self._batch_size(lpet_images))
                # train_mse.update(mse, self._batch_size(lpet_images))
                # train_nrmse.update(nrmse, self._batch_size(lpet_images))
                # train_ssim.update(ssim, self._batch_size(lpet_images))
                # compute eval criterion
                log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}], " \
                          f"Loss: {arnet_loss.item():.4f}, content_loss_prenet: {content_loss_prenet.item():.4f},"\
                          f"content_loss_arnet: {content_loss_arnet.item():.4f}, adv_loss: {adv_loss.item():.4f},"\
                          f"train_psnr: {psnr:.4f}, train_mse: {mse:.4f}, train_nrmse: {nrmse:.4f}, train_ssim: {ssim:.4f}"  #,
                # Epoch [{int(epoch) + 1}/{self.num_epochs}],
                # print(log_message)
                # with open(log_file, 'a') as f:
                #     f.write(log_message + '\n')

                # log stats, params and images
                logger.info(log_message)
                self._log_stats('train', train_loss.avg,arnet_loss.item(),content_loss_prenet.item(),content_loss_arnet.item(),adv_loss.item(),psnr,mse,nrmse,ssim)
                self._log_params()
                self._log_images(lpet_images, spet_images, fake_images, preliminary_predictions, real_residual,  estimated_residual, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False
    def validate(self):
        logger.info('Validating...')
        val_loss = RunningAverage()
        val_loss_adv = RunningAverage()
        val_loss_pre = RunningAverage()
        val_loss_ar = RunningAverage()
        val_psnr = RunningAverage()
        val_ssim = RunningAverage()
        val_mse = RunningAverage()
        val_nrmse = RunningAverage()
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['val']), total=len(self.loader['val']),
                desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration+1
                val_lpet_images, val_spet_images, weight = self._split_training_batch(t)

                # Forward pass and compute losses
                val_preliminary_predictions = self.generate(val_lpet_images)
                val_estimated_residual = self.refine(val_preliminary_predictions)
                # val_estimated_residual = val_preliminary_predictions * val_rectified_parameters
                val_rectified_spet_like_images = val_preliminary_predictions + val_estimated_residual

                val_real_images = val_spet_images
                val_fake_images = val_rectified_spet_like_images
                  # print(real_images.shape)

                val_advnet_real_output = self.discriminate(torch.cat((val_lpet_images, val_real_images), dim=1))
                val_advnet_fake_output = self.discriminate(torch.cat((val_lpet_images, val_fake_images), dim=1))

                val_adv_real_loss = torch.mean((val_advnet_real_output - 1) ** 2)
                val_adv_fake_loss = torch.mean(val_advnet_fake_output ** 2)
                # print(val_adv_real_loss,a_val_adv_real_loss,val_adv_fake_loss,a_val_adv_fake_loss)
                val_adv_loss = val_adv_real_loss + val_adv_fake_loss
                val_loss_adv.update(val_adv_loss,self._batch_size(val_lpet_images))

                  # advnet_loss_fake = criterion_adv(advnet_fake_output, fake_labels)
                  # advnet_loss = advnet_loss_real + advnet_loss_fake

                  # AR-Net: Train AR-Net with content and adversarial losses
                val_content_loss_prenet = self.lambda_content_prenet * self.content_loss(val_preliminary_predictions, val_spet_images)
                val_loss_pre.update(val_content_loss_prenet,self._batch_size(val_lpet_images))
                  # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))

                val_real_residual = val_spet_images - val_preliminary_predictions
                val_content_loss_arnet = self.lambda_content_arnet * self.content_loss(val_real_residual, val_estimated_residual)
                val_loss_ar.update(val_content_loss_arnet, self._batch_size(val_lpet_images))
                  # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )


                val_arnet_loss = val_adv_loss + val_content_loss_prenet + val_content_loss_arnet
                val_loss.update(val_arnet_loss.item(),self._batch_size(val_lpet_images))
                v_psnr = self.psnr(val_fake_images,val_real_images)
                val_psnr.update(v_psnr,self._batch_size(val_lpet_images))
                v_mse = self.mse(val_fake_images,val_real_images)
                val_mse.update(v_mse,self._batch_size(val_lpet_images))
                v_nrmse = NRMSE(val_fake_images,val_real_images)
                val_nrmse.update(v_nrmse,self._batch_size(val_lpet_images))
                v_ssim = self.ssim(val_fake_images,val_real_images)
                val_ssim.update(v_ssim,self._batch_size(val_lpet_images))

            logger.info(f"Validation Loss: {val_loss.avg:.4f}, val_psnr: {val_psnr.avg}, val_mse: {val_mse.avg}, val_nrmse: {val_nrmse.avg}, val_ssim: {val_ssim.avg}")  # Print the average validation loss
            self._log_stats('validation', val_loss.avg, val_arnet_loss.item(), val_loss_pre.avg,
                            val_loss_ar.avg,val_loss_adv.avg, val_psnr.avg, val_mse.avg, val_nrmse.avg, val_ssim.avg)
            self._log_images(val_lpet_images, val_spet_images, val_fake_images, val_preliminary_predictions,
                             val_real_residual,val_estimated_residual, 'validaion_')
        return {"val_loss": val_loss.avg, "val_psnr": val_psnr.avg, "val_mse": val_mse.avg, "val_nrmse": val_nrmse.avg, "val_ssim": val_ssim.avg}
    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best
    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def save_joint_checkpoint(self,state, is_best, checkpoint_dir,model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name+'_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name+'_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        if isinstance(self.generate,torch.nn.DataParallel):
            state_dict = self.generate.module.state_dict()
            refine_state_dict = self.refine.module.state_dict()
            dis_state_dict = self.discriminate.module.state_dict()
        else:
            state_dict = self.generate.state_dict()
            refine_state_dict = self.refine.state_dict()
            dis_state_dict = self.discriminate.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")
        #save model state
        self.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_pre.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)
        #save refine model state
        self.save_joint_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': refine_state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_refine.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir,model_name='refine')
        # save discriminal model state
        self.save_joint_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': dis_state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_disc.state_dict(),
        },is_best,  checkpoint_dir=self.checkpoint_dir, model_name='disc')

    def _log_lr(self):
        lr = self.optimizer_pre.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)
    def _log_stats(self, phase, loss_avg, loss,pre_loss,res_loss, disc_loss, psnr,mse,nrmse,ssim):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss': loss,
            f'{phase}_disc_loss': disc_loss,
            f'{phase}_pre_loss': pre_loss,
            f'{phase}_res_loss': res_loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse': mse,
            f'{phase}_nrmse': nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.generate.named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual':residual,
            'estimated_res':estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                # print('1')
                return tuple([_move_to_device(x) for x in input])
            else:
                # print(input)
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight


def create_ARGAN_3d_trainer_residual_refine(config):
    # Get the model we need
    generate_model = define_G(**config['generator'])
    # generate_model = get_model(config['generator'])
    refine_model = get_model(config['refine_model'])
    discriminate_model = get_model(config['discriminator'])
    # Get the device
    device = torch.device(config['device'])
    logger.info(f"Sending the model to '{config['device']}'")
    # device_ids = [0,1]
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        generate_model = nn.DataParallel(generate_model)#, device_ids=device_ids)
        refine_model = nn.DataParallel(refine_model)#, device_ids=device_ids)
        discriminate_model = nn.DataParallel(discriminate_model)#, device_ids=device_ids)

        # model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    # Put the model on the device
    generate_model = generate_model.to(device)
    refine_model = refine_model.to(device)
    discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()
    criterion_content = L1Loss()
    # Create evaluation metrics
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)

    optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']
    # lambda_content_prenet = trainer_config['lambda_content_prenet']
    # lambda_content_arnet = trainer_config['lambda_content_arnet']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return ARGANTrainer_residual_refine(
                         generate_model=generate_model,
                         refine_model=refine_model,
                         dis_model=discriminate_model,
                         optimizer_pre=optimizer_pre,
                        #  lr_scheduler_pre=lr_scheduler_pre,
                         optimizer_refine=optimizer_refine,
                        #  lr_scheduler_refine=lr_scheduler_refine,
                         optimizer_disc=optimizer_disc,
                         adv_loss = criterion_adv,
                         content_loss = criterion_content,
                         psnr = psnr,
                         mse = mse,
                         ssim = ssim,
                         tensorboard_formatter=tensorboard_formatter,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         lr = lr,
                         **trainer_config)



class ARGANTrainer_residual_refine:
    def __init__(self, generate_model, refine_model,dis_model, optimizer_pre,optimizer_disc,optimizer_refine,
                 lr, adv_loss, content_loss,psnr,
                 mse,ssim,device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.generate = generate_model
        self.refine = refine_model
        self.discriminate = dis_model
        self.optimizer_pre = optimizer_pre
        # self.scheduler_pre = lr_scheduler_pre
        self.optimizer_refine = optimizer_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizer_disc = optimizer_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generate, self.optimizer_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generate, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]
    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0,self.max_num_epochs)
        for epoch in tqdm.tqdm(
                    enumerate(epoch_list), total=self.max_num_epochs,
                    desc='Train epoch==%d' % self.num_epochs, ncols=80,
                    leave=False):

            lr = self.check_lr(epoch[1],decay_epoch=30)

            for param_group in self.optimizer_pre.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_refine.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_disc.param_groups:
                param_group['lr'] = lr


            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self,epoch,decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse= RunningAverage()
        # train_nrmse= RunningAverage()


        self.generate.train()
        self.refine.train()
        self.discriminate.train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations,self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images , spet_images, weight = self._split_training_batch(t)

            # PreNet: Generate preliminary predictions
            preliminary_predictions = self.generate(lpet_images)
            # AR-Net: Generate rectified parameters and estimated residual
            # rectified_input = (preliminary_predictions - lpet_images)/(preliminary_predictions + 1e-10)
            rectified_input = preliminary_predictions - lpet_images
            # rectified_input = (preliminary_predictions - lpet_images) / (lpet_images + 1e-10)
            estimated_residual= self.refine(rectified_input)
            # estimated_residual = rectified_parameters
            # estimated_residual = preliminary_predictions * rectified_parameters
            # estimated_residual = lpet_images * rectified_parameters

            # Combine estimated residual with preliminary predictions
            # rectified_spet_like_images = lpet_images+(preliminary_predictions + 1e-10)*estimated_residual
            # rectified_spet_like_images = lpet_images + (lpet_images + 1e-10) * estimated_residual
            rectified_spet_like_images = preliminary_predictions + estimated_residual
            # rectified_spet_like_images = lpet_images + (preliminary_predictions + 1e-10)*estimated_residual

            # print(rectified_spet_like_images.shape)

            # AdvNet: Train AdvNet with real and fake image pairs
            real_images = spet_images
            fake_images = rectified_spet_like_images
            # print(real_images,fake_images)
            # logger.info(f'image size:{lpet_images.shape,fake_images.shape,real_images.shape}')
            # print(real_images.shape)

            advnet_real_output = self.discriminate(torch.cat((lpet_images,real_images),dim=1))
            advnet_fake_output = self.discriminate(torch.cat((lpet_images,fake_images),dim=1)) # (fake_images.detach())

            adv_real_loss = torch.mean((advnet_real_output-1)**2)
            adv_fake_loss = torch.mean(advnet_fake_output**2)

            adv_loss = adv_real_loss+adv_fake_loss
            # print(f'adv_loss: {adv_loss}')

            # AR-Net: Train AR-Net with content and adversarial losses
            content_loss_prenet = self.lambda_content_prenet * self.content_loss(preliminary_predictions, spet_images)
            # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
            # print(f'content_loss_prenet{content_loss_prenet}')

            # real_residual = spet_images - lpet_images
            real_residual = spet_images - preliminary_predictions
            # real_residual = (spet_images - lpet_images) / (lpet_images + 1e-10)
            # real_residual = (spet_images - lpet_images)/(preliminary_predictions + 1e-10)
            content_loss_arnet = self.lambda_content_arnet * self.content_loss(real_residual, estimated_residual)
            # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )
            # print(f'content_loss_arnet{content_loss_arnet}')
            # content_loss_final = self.content_loss(spet_images, fake_images)



            arnet_loss =adv_loss + content_loss_prenet + content_loss_arnet
            train_loss.update(arnet_loss.item(),self._batch_size(lpet_images))
            # print(f'arnet_loss{arnet_loss}')
            psnr = self.psnr(fake_images,real_images)
#             train_psnr.update(psnr,self._batch_size(lpet_images))
            mse = self.mse(fake_images,real_images)
#             train_mse.update(mse,self._batch_size(lpet_images))
            nrmse = NRMSE(fake_images,real_images)
#             train_nrmse.update(nrmse,self._batch_size(lpet_images))
            ssim = self.ssim(fake_images,real_images)
#             train_ssim.update(ssim,self._batch_size(lpet_images))

            self.optimizer_pre.zero_grad()
            self.optimizer_refine.zero_grad()
            self.optimizer_disc.zero_grad()
            # print('finish zero grad')
            arnet_loss.backward()
            # print('finish backward')
            self.optimizer_pre.step()
            self.optimizer_refine.step()
            self.optimizer_disc.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the models in eval mode
                self.generate.eval()
                self.refine.eval()
                self.discriminate.eval()
                # evaluate on validation set
                val_result = self.validate()
                # set the model back to training mode
                self.generate.train()
                self.refine.train()
                self.discriminate.train()

                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss'])
                # is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')

            if self.num_iterations % self.log_after_iters == 0:
                # train_loss.update(arnet_loss.item(), self._batch_size(lpet_images))
                # train_psnr.update(psnr, self._batch_size(lpet_images))
                # train_mse.update(mse, self._batch_size(lpet_images))
                # train_nrmse.update(nrmse, self._batch_size(lpet_images))
                # train_ssim.update(ssim, self._batch_size(lpet_images))
                # compute eval criterion
                log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}], " \
                          f"Loss: {arnet_loss.item():.4f}, content_loss_prenet: {content_loss_prenet.item():.4f},"\
                          f"content_loss_arnet: {content_loss_arnet.item():.4f}, adv_loss: {adv_loss.item():.4f},"\
                          f"train_psnr: {psnr:.4f}, train_mse: {mse:.4f}, train_nrmse: {nrmse:.4f}, train_ssim: {ssim:.4f}"  #,
                # Epoch [{int(epoch) + 1}/{self.num_epochs}],
                # print(log_message)
                # with open(log_file, 'a') as f:
                #     f.write(log_message + '\n')

                # log stats, params and images
                logger.info(log_message)
                self._log_stats('train', train_loss.avg,arnet_loss.item(),content_loss_prenet.item(),content_loss_arnet.item(),adv_loss.item(),psnr,mse,nrmse,ssim)
                self._log_params()
                self._log_images(lpet_images, spet_images, fake_images, preliminary_predictions, real_residual,  estimated_residual, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False
    def validate(self):
        logger.info('Validating...')
        val_loss = RunningAverage()
        val_loss_adv = RunningAverage()
        val_loss_pre = RunningAverage()
        val_loss_ar = RunningAverage()
        val_psnr = RunningAverage()
        val_ssim = RunningAverage()
        val_mse = RunningAverage()
        val_nrmse = RunningAverage()
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['val']), total=len(self.loader['val']),
                desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration+1
                val_lpet_images, val_spet_images, weight = self._split_training_batch(t)

                # Forward pass and compute losses
                val_preliminary_predictions = self.generate(val_lpet_images)
                # val_rectified_input= (val_preliminary_predictions - val_lpet_images) / (val_preliminary_predictions + 1e-10)
                # val_rectified_input = (val_preliminary_predictions - val_lpet_images) / (val_lpet_images + 1e-10)
                val_rectified_input = val_preliminary_predictions - val_lpet_images
                val_estimated_residual = self.refine(val_rectified_input)
                # val_estimated_residual = val_rectified_parameters
                # val_estimated_residual = val_lpet_images * val_rectified_parameters
                # val_estimated_residual = val_preliminary_predictions * val_rectified_parameters
                # val_rectified_spet_like_images = val_lpet_images + val_estimated_residual*(val_preliminary_predictions + 1e-10)
                # val_rectified_spet_like_images = val_lpet_images + val_estimated_residual * (val_lpet_images+ 1e-10)
                val_rectified_spet_like_images = val_preliminary_predictions + val_estimated_residual
                # val_rectified_spet_like_images = val_lpet_images + (val_preliminary_predictions + 1e-10)*val_estimated_residual

                val_real_images = val_spet_images
                val_fake_images = val_rectified_spet_like_images
                  # print(real_images.shape)

                val_advnet_real_output = self.discriminate(torch.cat((val_lpet_images, val_real_images), dim=1))
                val_advnet_fake_output = self.discriminate(torch.cat((val_lpet_images, val_fake_images), dim=1))

                val_adv_real_loss = torch.mean((val_advnet_real_output - 1) ** 2)
                val_adv_fake_loss = torch.mean(val_advnet_fake_output ** 2)
                # print(val_adv_real_loss,a_val_adv_real_loss,val_adv_fake_loss,a_val_adv_fake_loss)
                val_adv_loss = val_adv_real_loss + val_adv_fake_loss
                val_loss_adv.update(val_adv_loss,self._batch_size(val_lpet_images))

                  # advnet_loss_fake = criterion_adv(advnet_fake_output, fake_labels)
                  # advnet_loss = advnet_loss_real + advnet_loss_fake

                  # AR-Net: Train AR-Net with content and adversarial losses
                val_content_loss_prenet = self.lambda_content_prenet * self.content_loss(val_preliminary_predictions, val_spet_images)
                val_loss_pre.update(val_content_loss_prenet,self._batch_size(val_lpet_images))
                  # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))

                # val_real_residual = val_spet_images - val_lpet_images
                val_real_residual = val_spet_images - val_preliminary_predictions
                # val_real_residual = (val_spet_images - val_lpet_images)/(val_preliminary_predictions + 1e-10)
                # val_real_residual = (val_spet_images - val_lpet_images) / (val_lpet_images + 1e-10)
                val_content_loss_arnet = self.lambda_content_arnet * self.content_loss(val_real_residual, val_estimated_residual)
                val_loss_ar.update(val_content_loss_arnet, self._batch_size(val_lpet_images))
                  # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )


                val_arnet_loss = val_adv_loss + val_content_loss_prenet + val_content_loss_arnet
                val_loss.update(val_arnet_loss.item(),self._batch_size(val_lpet_images))
                v_psnr = self.psnr(val_fake_images,val_real_images)
                val_psnr.update(v_psnr,self._batch_size(val_lpet_images))
                v_mse = self.mse(val_fake_images,val_real_images)
                val_mse.update(v_mse,self._batch_size(val_lpet_images))
                v_nrmse = NRMSE(val_fake_images,val_real_images)
                val_nrmse.update(v_nrmse,self._batch_size(val_lpet_images))
                v_ssim = self.ssim(val_fake_images,val_real_images)
                val_ssim.update(v_ssim,self._batch_size(val_lpet_images))

            logger.info(f"Validation Loss: {val_loss.avg:.4f}, val_psnr: {val_psnr.avg}, val_mse: {val_mse.avg}, val_nrmse: {val_nrmse.avg}, val_ssim: {val_ssim.avg}")  # Print the average validation loss
            self._log_stats('validation', val_loss.avg, val_arnet_loss.item(), val_loss_pre.avg,
                            val_loss_ar.avg,val_loss_adv.avg, val_psnr.avg, val_mse.avg, val_nrmse.avg, val_ssim.avg)
            self._log_images(val_lpet_images, val_spet_images, val_fake_images, val_preliminary_predictions,
                             val_real_residual,val_estimated_residual, 'validaion_')
        return {"val_loss": val_loss.avg, "val_psnr": val_psnr.avg, "val_mse": val_mse.avg, "val_nrmse": val_nrmse.avg, "val_ssim": val_ssim.avg}
    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best
    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def save_joint_checkpoint(self,state, is_best, checkpoint_dir,model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name+'_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name+'_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        if isinstance(self.generate,torch.nn.DataParallel):
            state_dict = self.generate.module.state_dict()
            refine_state_dict = self.refine.module.state_dict()
            dis_state_dict = self.discriminate.module.state_dict()
        else:
            state_dict = self.generate.state_dict()
            refine_state_dict = self.refine.state_dict()
            dis_state_dict = self.discriminate.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")
        #save model state
        self.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_pre.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)
        #save refine model state
        self.save_joint_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': refine_state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_refine.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir,model_name='refine')
        # save discriminal model state
        self.save_joint_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': dis_state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_disc.state_dict(),
        },is_best,  checkpoint_dir=self.checkpoint_dir, model_name='disc')

    def _log_lr(self):
        lr = self.optimizer_pre.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)
    def _log_stats(self, phase, loss_avg, loss,pre_loss,res_loss, disc_loss, psnr,mse,nrmse,ssim):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss': loss,
            f'{phase}_disc_loss': disc_loss,
            f'{phase}_pre_loss': pre_loss,
            f'{phase}_res_loss': res_loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse': mse,
            f'{phase}_nrmse': nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.generate.named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual':residual,
            'estimated_res':estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                # print('1')
                return tuple([_move_to_device(x) for x in input])
            else:
                # print(input)
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

def create_DRFARGAN_3d_trainer(config):
    drf_list = ['Full_dose']
    generate_models = {drf: define_G(**config['generator']) for drf in drf_list}
    refine_models = {drf: get_model(config['refine_model']) for drf in drf_list}
    discriminate_models = {drf: get_model(config['discriminator']) for drf in drf_list}

    # Get the model we need
    # Get the device
    device = torch.device(config['device'])
    # device_ids = [0,1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        for drf, model in generate_models.items():
            generate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in refine_models.items():
            refine_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

    else:
        for drf, model in generate_models.items():
            generate_models[drf] = model.to(device)

        for drf, model in refine_models.items():
            refine_models[drf] = model.to(device)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = model.to(device)

    # generate_model = torch.nn.DataParallel(generate_model, device_ids=device_ids)
    # refine_model = torch.nn.DataParallel(refine_model, device_ids=device_ids)
    # discriminate_model = torch.nn.DataParallel(discriminate_model, device_ids=device_ids)

    logger.info(f"Sending the model to '{config['device']}'")
    # # Put the model on the device
    # generate_model = generate_model.to(device)
    # refine_model = refine_model.to(device)
    # discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()#MSELoss() #CrossEntropyLoss() #
    criterion_content = L1Loss()
    # Create evaluation metrics
    # psnr = torch.nn.DataParallel(PeakSignalNoiseRatio(), device_ids=device_ids)
    # mse = torch.nn.DataParallel(MeanSquaredError(), device_ids=device_ids)
    # ssim = torch.nn.DataParallel(StructuralSimilarityIndexMeasure(), device_ids=device_ids)
    # torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)
    optimizers_pre = {drf: create_optimizer(config['optimizer'], generate_models[drf]) for drf in drf_list}
    optimizers_refine = {drf: create_optimizer(config['optimizer'], refine_models[drf]) for drf in drf_list}
    optimizers_disc = {drf: create_optimizer(config['optimizer'], discriminate_models[drf]) for drf in drf_list}


    # optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    # optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    # optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)
    print('drf_list',drf_list)

    return DRFARGANTrainer(
                         generate_models=generate_models,
                         refine_models=refine_models,
                         dis_models=discriminate_models,
                         optimizers_pre=optimizers_pre,
                        #  lr_scheduler_pre=lr_scheduler_pre,
                         optimizers_refine=optimizers_refine,
                        #  lr_scheduler_refine=lr_scheduler_refine,
                         optimizers_disc=optimizers_disc,
                         adv_loss = criterion_adv,
                         content_loss = criterion_content,
                         psnr = psnr,
                         mse = mse,
                         ssim = ssim,
                         tensorboard_formatter=tensorboard_formatter,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         lr = lr,
                         drf_list = drf_list,
                         **trainer_config)




class DRFARGANTrainer:
    def __init__(self, generate_models, refine_models,dis_models, optimizers_pre,optimizers_disc,optimizers_refine,
                 lr, drf_list,adv_loss, content_loss,psnr,mse,ssim,device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.generators = generate_models
        self.refiners = refine_models
        self.discriminators = dis_models
        self.optimizers_pre = optimizers_pre
        # self.scheduler_pre = lr_scheduler_pre
        self.optimizers_refine = optimizers_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizers_disc = optimizers_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.drf_list = drf_list
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generators, self.optimizers_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generators, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]
    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0,self.max_num_epochs)
        for epoch in tqdm.tqdm(
                    enumerate(epoch_list), total=self.max_num_epochs,
                    desc='Train epoch==%d' % self.num_epochs, ncols=80,
                    leave=False):

            lr = self.check_lr(epoch[1],decay_epoch=25)
            for drf in self.drf_list:
                for param_group in self.optimizers_pre[drf].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizers_refine[drf].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizers_disc[drf].param_groups:
                    param_group['lr'] = lr


            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self,epoch,decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse = RunningAverage()
        # train_nrmse = RunningAverage()


        for model_name, gen in self.generators.items():
            gen.train()
            self.refiners[model_name].train()
            self.discriminators[model_name].train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations,self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images , target_images, weight = self._split_training_batch(t)
            generate_images = {}
            
            input_data = lpet_images
            input_images = {}
            total_loss = {}
            disc_loss = {}
            prenet_loss = {}
            residual_loss = {}
            psnr_drf = {}
            ssim_drf = {}
            mse_drf = {}
            nrmse_drf = {}
            pre_predictions = {}
            real_residuals = {}  
            estimated_residuals = {}
            all_model_loss = 0
            for model_name, gen in self.generators.items():
                # print(model_name)
                input_images[model_name] = input_data
                # print('gen input',input_data.shape)
                preliminary_predictions = gen(input_data)
#                 print('gen output',preliminary_predictions.shape)
                rectified_parameters = self.refiners[model_name](preliminary_predictions)
                estimated_residuals[model_name] = preliminary_predictions * rectified_parameters
                rectified_spet_like_images = preliminary_predictions+estimated_residuals[model_name]
                pre_predictions[model_name] = preliminary_predictions
#                 print('residual')
                generate_images[model_name] = rectified_spet_like_images
                real_residuals[model_name] = target_images[model_name] - preliminary_predictions

#                 print('start loss')
                advnet_real_output = self.discriminators[model_name](torch.cat((input_data,target_images[model_name]),dim=1).detach())
                advnet_fake_output = self.discriminators[model_name](torch.cat((input_data,generate_images[model_name]),dim=1).detach()) # (fake_images.detach())

                adv_real_loss = torch.mean((advnet_real_output-1)**2)
                adv_fake_loss = torch.mean(advnet_fake_output**2)

                disc_loss[model_name] = adv_real_loss+adv_fake_loss

                # AR-Net: Train AR-Net with content and adversarial losses
                prenet_loss[model_name] = self.lambda_content_prenet * self.content_loss(preliminary_predictions, target_images[model_name])
                # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
                # print(f'content_loss_prenet{content_loss_prenet}')


                residual_loss[model_name] = self.lambda_content_arnet * self.content_loss(real_residuals[model_name], estimated_residuals[model_name])
             
                total_loss[model_name] = disc_loss[model_name] + prenet_loss[model_name] + residual_loss[model_name]
                # total_loss[model_name] = arnet_loss
                all_model_loss += total_loss[model_name]
                # print(total_loss)
                # print(arnet_loss)
                # print(all_model_loss)

                
                # print(f'arnet_loss{arnet_loss}')
                psnr = self.psnr(generate_images[model_name],target_images[model_name])
                psnr_drf[model_name] = psnr
                # train_psnr.update(psnr,self._batch_size(lpet_images))
                mse = self.mse(generate_images[model_name],target_images[model_name])
                mse_drf[model_name] = mse
    #             train_mse.update(mse,self._batch_size(lpet_images))
                nrmse = NRMSE(generate_images[model_name],target_images[model_name])
                nrmse_drf[model_name] = nrmse
    #             train_nrmse.update(nrmse,self._batch_size(lpet_images))
                ssim = self.ssim(generate_images[model_name],target_images[model_name])
                ssim_drf[model_name] = ssim
    #             train_ssim.update(ssim,self._batch_size(lpet_images))
                input_data = generate_images[model_name].detach()
            # train_loss.update(all_model_loss, self._batch_size(input_data))

            # for model_name, optimize_pre in self.optimizers_pre.items():
                self.optimizers_pre[model_name].zero_grad()
                self.optimizers_refine[model_name].zero_grad()
                self.optimizers_disc[model_name].zero_grad()
                # print('finish zero grad')
                # all_model_loss.backward(retain_graph=True)
                total_loss[model_name].backward()
                # print('finish backward')
                self.optimizers_pre[model_name].step()
                self.optimizers_refine[model_name].step()
                self.optimizers_disc[model_name].step()
            train_loss.update(all_model_loss, self._batch_size(input_data))




            if self.num_iterations % self.validate_after_iters == 0:
                print('start val')
                # set the models in eval mode
                for model_name, gen in self.generators.items():
                    gen.eval()
                    self.refiners[model_name].eval()
                    self.discriminators[model_name].eval()
                # evaluate on validation set
                val_result = self.validate()
                for model_name, gen in self.generators.items():
                    # set the model back to training mode
                    gen.train()
                    self.refiners[model_name].train()
                    self.discriminators[model_name].train()

                    # # adjust learning rate if necessary
                    # if isinstance(self.scheduler, ReduceLROnPlateau):
                    #     self.scheduler.step(eval_score)
                    # else:
                    #     self.scheduler.step()

                    # log current learning rate in tensorboard
                    self._log_lr(model_name)
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss']['Full_dose'])
                    # is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')


            if self.num_iterations % self.log_after_iters == 0:
                for drf in self.drf_list:
                    # compute eval criterion
                    log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}],Total_Loss: {all_model_loss} , {drf} as the input: " \
                                f"Loss: {total_loss[drf].item():.4f}, content_loss_prenet: {prenet_loss[drf].item():.4f},"\
                                f"content_loss_arnet: {residual_loss[drf].item():.4f}, adv_loss: {disc_loss[drf].item():.4f},"\
                                f"train_psnr: {psnr_drf[drf]:.4f}, train_mse: {mse_drf[drf]:.4f}, train_nrmse: {nrmse_drf[drf]:.4f}, train_ssim: {ssim_drf[drf]:.4f}"  #,
                        # log stats, params and images
                    logger.info(log_message)
                    # print('log_stats',drf)
                    self._log_stats(f'train_{drf}', all_model_loss ,total_loss[drf].item(),prenet_loss[drf].item(),residual_loss[drf].item(),disc_loss[drf].item(),psnr_drf[drf],mse_drf[drf],nrmse_drf[drf],ssim_drf[drf])
                    # print('log_params', drf)
                    self._log_params(drf)
                    # print('log_images', drf)
                    self._log_images(input_images[drf], target_images[drf], generate_images[drf], pre_predictions[drf], real_residuals[drf],  estimated_residuals[drf], f'train_{drf}_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False
    def validate(self):
        logger.info('Validating...')
        val_loss = {}
        val_disc_loss = {}
        val_prenet_loss = {}
        val_residual_loss = {}
        val_psnr_dict = {}
        val_ssim_dict = {}
        val_mse_dict = {}
        val_nrmse_dict = {}

        for drf in self.drf_list:
            val_loss[drf] = 0
            val_prenet_loss[drf] = 0
            val_residual_loss[drf] = 0
            val_disc_loss[drf] = 0
            val_psnr_dict[drf] = 0
            val_ssim_dict[drf] = 0
            val_mse_dict[drf] = 0
            val_nrmse_dict[drf] = 0
        val_loss_all_model = 0
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['val']), total=len(self.loader['val']),
                desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration+1

                val_lpet_images , val_target_images, val_weight = self._split_training_batch(t)
                val_generate_images = {}
                val_input_data = val_lpet_images
                val_total_loss = {}
                val_input_images = {}
                val_pre_predictions = {}
                val_real_residuals = {}  
                val_estimated_residuals = {}
                val_all_model_loss = 0
                for model_name, val_gen in self.generators.items():
                    print('val',model_name)
                    val_input_images[model_name] = val_input_data
                    val_preliminary_predictions = val_gen(val_input_data)
                    val_rectified_parameters = self.refiners[model_name](val_preliminary_predictions)
                    val_estimated_residual = val_preliminary_predictions * val_rectified_parameters
                    val_rectified_spet_like_images = val_preliminary_predictions+val_estimated_residual
                    val_pre_predictions[model_name] = val_preliminary_predictions

                    val_estimated_residuals[model_name] = val_estimated_residual
                    val_target = val_target_images[model_name]
                    val_generate_images[model_name] = val_rectified_spet_like_images

                    val_advnet_real_output = self.discriminators[model_name](torch.cat((val_input_data,val_target),dim=1))
                    val_advnet_fake_output = self.discriminators[model_name](torch.cat((val_input_data,val_generate_images[model_name]),dim=1)) # (fake_images.detach())

                    val_adv_real_loss = torch.mean((val_advnet_real_output-1)**2)
                    val_adv_fake_loss = torch.mean(val_advnet_fake_output**2)

                    val_adv_loss = val_adv_real_loss+val_adv_fake_loss
                    val_disc_loss[model_name] += val_adv_loss
                    # AR-Net: Train AR-Net with content and adversarial losses
                    val_content_loss_prenet = self.lambda_content_prenet * self.content_loss(val_preliminary_predictions, val_target)
                    val_prenet_loss[model_name] += val_content_loss_prenet
                    # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
                    # print(f'content_loss_prenet{content_loss_prenet}')

                    val_real_residual = val_target - val_preliminary_predictions
                    val_real_residuals[model_name] = val_real_residual
                    val_content_loss_arnet = self.lambda_content_arnet * self.content_loss(val_real_residual, val_estimated_residual)
                    val_residual_loss[model_name] += val_content_loss_arnet
                
                    val_arnet_loss =val_adv_loss + val_content_loss_prenet + val_content_loss_arnet
                    val_all_model_loss += val_arnet_loss

                    val_loss[model_name] += val_arnet_loss

                    # print(f'arnet_loss{arnet_loss}')
                    val_psnr = self.psnr(val_generate_images[model_name],val_target)
                    val_psnr_dict[model_name] += val_psnr
                    val_mse = self.mse(val_generate_images[model_name],val_target)
                    val_mse_dict[model_name] += val_mse
                    val_nrmse = NRMSE(val_generate_images[model_name],val_target)
                    val_nrmse_dict[model_name] += val_nrmse
                    val_ssim = self.ssim(val_generate_images[model_name],val_target)
                    val_ssim_dict[model_name] += val_ssim
                    val_input_data = val_generate_images[model_name].detach()
                val_loss_all_model += val_all_model_loss
            num = len(self.loader['val'])
            dicts_list = [val_loss, val_disc_loss, val_prenet_loss, val_residual_loss, val_psnr_dict, val_ssim_dict,
                              val_mse_dict, val_nrmse_dict]
            for my_dict in dicts_list:
                for key in my_dict:
                    my_dict[key] = my_dict[key]/num

            for drf in self.drf_list:
                num = len(self.loader['val'])
                    # compute eval criterion
                log_message = f"Total_Loss: {val_loss_all_model/num} , {drf} as the target: " \
                                f"Loss: {val_loss[drf]:.4f}, content_loss_prenet: {val_prenet_loss[drf]:.4f},"\
                                f"content_loss_arnet: {val_residual_loss[drf]:.4f}, adv_loss: {val_disc_loss[drf]:.4f},"\
                                f"val_psnr: {val_psnr_dict[drf]:.4f}, val_mse: {val_mse_dict[drf]:.4f}, val_nrmse: {val_nrmse_dict[drf]:.4f}, val_ssim: {val_ssim_dict[drf]:.4f}"  #,
                        # log stats, params and images
                logger.info(log_message)
                self._log_stats(f'val_{drf}', val_loss_all_model/num,val_loss[drf],val_prenet_loss[drf],val_residual_loss[drf],val_disc_loss[drf],val_psnr_dict[drf],val_mse_dict[drf],val_nrmse_dict[drf],val_ssim_dict[drf])
                self._log_images(val_input_images[drf], val_target_images[self.drf_list[self.drf_list.index(drf)]], val_generate_images[drf], val_pre_predictions[drf], val_real_residuals[drf],  val_estimated_residuals[drf], f'val_{drf}_')

        return {"val_all_model_loss":val_loss_all_model/num,"val_loss": val_loss, "val_psnr": val_psnr_dict, "val_mse": val_mse_dict, "val_nrmse": val_nrmse_dict, "val_ssim": val_ssim_dict}
    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best
    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def save_joint_checkpoint(self,state, is_best, checkpoint_dir,model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name+'_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name+'_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)
    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        state_dict = {}
        refine_state_dict = {}
        dis_state_dict = {}
        for model_name, gen in self.generators.items():
            if isinstance(self.generators[model_name],torch.nn.DataParallel):
                state_dict[model_name] = self.generators[model_name].module.state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].module.state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].module.state_dict()
            else:
                state_dict[model_name] = self.generators[model_name].state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].state_dict()

            last_file_path = os.path.join(self.checkpoint_dir,model_name, 'last_checkpoint.pytorch')
            logger.info(f"Saving checkpoint to '{last_file_path}'")
            #save model state
            self.save_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_pre[model_name].state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir,model_name))
            #save refine model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': refine_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_refine[model_name].state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir,model_name),model_name='refine')
            # save discriminal model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': dis_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_disc[model_name].state_dict(),
            },is_best,  checkpoint_dir=os.path.join(self.checkpoint_dir,model_name), model_name='disc')

    def _log_lr(self,model_name):
        lr = self.optimizers_pre[model_name].param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)
    def _log_stats(self, phase, all_model_loss, loss,pre_loss,res_loss, disc_loss, psnr,mse,nrmse,ssim):
        tag_value = {
            f'{phase}_loss_all_model': all_model_loss,
            f'{phase}_loss': loss,
            f'{phase}_disc_loss': disc_loss,
            f'{phase}_pre_loss': pre_loss,
            f'{phase}_res_loss': res_loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse':mse,
            f'{phase}_nrmse':nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self,drf):
        logger.info('Logging model parameters and gradients')
        for name, value in self.generators[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        for name, value in self.refiners[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        for name, value in self.discriminators[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual':residual,
            'estimated_res':estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        # print(t)
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    input = input.to(self.device)
                if isinstance(input,dict):
                    for name, data in input.items():
                        input[name] = input[name].to(self.device)
                # print(input)
                return input

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight


def create_DRFARGAN_3d_trainer_total_back(config):
    drf_list = [ 'drf_20',  'drf_4',  'Full_dose']
    generate_models = {drf: define_G(**config['generator']) for drf in drf_list}
    # generate_models = {drf: get_model(config['generator']) for drf in drf_list}
    refine_models = {drf: get_model(config['refine_model']) for drf in drf_list}
    discriminate_models = {drf: get_model(config['discriminator']) for drf in drf_list}

    # Get the model we need
    # Get the device
    device = torch.device(config['device'])
    # device_ids = [0,1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        for drf, model in generate_models.items():
            generate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in refine_models.items():
            refine_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

    else:
        for drf, model in generate_models.items():
            generate_models[drf] = model.to(device)

        for drf, model in refine_models.items():
            refine_models[drf] = model.to(device)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = model.to(device)

    logger.info(f"Sending the model to '{config['device']}'")
    # # Put the model on the device
    # generate_model = generate_model.to(device)
    # refine_model = refine_model.to(device)
    # discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()  # MSELoss() #CrossEntropyLoss() #
    criterion_content = L1Loss()
    # Create evaluation metrics
    # psnr = torch.nn.DataParallel(PeakSignalNoiseRatio(), device_ids=device_ids)
    # mse = torch.nn.DataParallel(MeanSquaredError(), device_ids=device_ids)
    # ssim = torch.nn.DataParallel(StructuralSimilarityIndexMeasure(), device_ids=device_ids)
    # torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)
    optimizer_pre = create_optimizer(config['optimizer'], generate_models)
    optimizer_refine =create_optimizer(config['optimizer'], refine_models)
    optimizer_disc = create_optimizer(config['optimizer'], discriminate_models)

    # optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    # optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    # optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return DRFARGANTrainer_total_back(
        drf_list = drf_list,
        generate_models=generate_models,
        refine_models=refine_models,
        dis_models=discriminate_models,
        optimizer_pre=optimizer_pre,
        #  lr_scheduler_pre=lr_scheduler_pre,
        optimizer_refine=optimizer_refine,
        #  lr_scheduler_refine=lr_scheduler_refine,
        optimizer_disc=optimizer_disc,
        adv_loss=criterion_adv,
        content_loss=criterion_content,
        psnr=psnr,
        mse=mse,
        ssim=ssim,
        tensorboard_formatter=tensorboard_formatter,
        device=config['device'],
        loaders=loaders,
        resume=resume,
        pre_trained=pre_trained,
        lr=lr,
        **trainer_config)




class DRFARGANTrainer_total_back:
    def __init__(self, drf_list,generate_models, refine_models, dis_models, optimizer_pre, optimizer_disc, optimizer_refine,
                 lr, adv_loss, content_loss, psnr,
                 mse, ssim, device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.drf_list = drf_list
        self.generators = generate_models
        self.refiners = refine_models
        self.discriminators = dis_models
        self.optimizer_pre = optimizer_pre
        # self.scheduler_pre = lr_scheduler_pre
        self.optimizer_refine = optimizer_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizer_disc = optimizer_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generators, self.optimizer_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generators, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0, self.max_num_epochs)
        for epoch in tqdm.tqdm(
                enumerate(epoch_list), total=self.max_num_epochs,
                desc='Train epoch==%d' % self.num_epochs, ncols=80,
                leave=False):

            lr = self.check_lr(epoch[1], decay_epoch=100)
            for param_group in self.optimizer_pre.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_refine.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_disc.param_groups:
                param_group['lr'] = lr

            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self, epoch, decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse = RunningAverage()
        # train_nrmse = RunningAverage()

        for model_name, gen in self.generators.items():
            gen.train()
            self.refiners[model_name].train()
            self.discriminators[model_name].train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations, self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images, target_images, weight = self._split_training_batch(t)

            pre_images, generate_images, estimate_residuals, real_residuals = self.forward_pass(lpet_images,target_images)

            loss = self.loss_calculate(pre_images, generate_images, estimate_residuals,target_images, real_residuals,lpet_images)
            train_loss.update(loss.item(),self._batch_size(lpet_images))

            psnr, ssim, mse, nrmse = self.evaluation_calculate(generate_images['Full_dose'],target_images['Full_dose'])

            self.optimizer_pre.zero_grad()
            self.optimizer_refine.zero_grad()
            self.optimizer_disc.zero_grad()
            loss.backward()
            self.optimizer_pre.step()
            self.optimizer_refine.step()
            self.optimizer_disc.step()

            if self.num_iterations % self.validate_after_iters == 0:
                print('start val')
                # set the models in eval mode
                for model_name, gen in self.generators.items():
                    gen.eval()
                    self.refiners[model_name].eval()
                    self.discriminators[model_name].eval()
                # evaluate on validation set
                val_result = self.validate()
                for model_name, gen in self.generators.items():
                    # set the model back to training mode
                    gen.train()
                    self.refiners[model_name].train()
                    self.discriminators[model_name].train()

                    # log current learning rate in tensorboard
                    self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss'])
                    # is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')

            if self.num_iterations % self.log_after_iters == 0:
                    # compute eval criterion
                    log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}],Total_Loss: {loss} ," \
                                f"train_psnr: {psnr:.4f}, train_mse: {mse:.4f}, train_nrmse: {nrmse:.4f}, train_ssim: {ssim:.4f}"  #,
                        # log stats, params and images
                    logger.info(log_message)
                    # print('log_stats',drf)
                    self._log_stats(f'train',train_loss.avg, loss.item(),psnr,mse,nrmse,ssim)
                    # print('log_params', drf)
                    self._log_params()
                    # print('log_images', drf)
                    self._log_images(lpet_images, target_images['Full_dose'], generate_images['Full_dose'], pre_images['Full_dose'], real_residuals['Full_dose'],  estimate_residuals['Full_dose'], f'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1
        return False

    def forward_pass(self,input_data,target):
        input = input_data
        generate_images = {}
        estimate_residuals = {}
        real_residuals = {}
        pre_images = {}
        for model_name, gen in self.generators.items():
            # print(model_name)
            # print('gen input',input_data.shape)
            preliminary_predictions = gen(input)
#             print('gen output')
            rectified_parameters = self.refiners[model_name](preliminary_predictions)
#             print('refine')
            estimated_residual = preliminary_predictions * rectified_parameters
            rectified_spet_like_images = preliminary_predictions + estimated_residual
#             print('residual')
            input = rectified_spet_like_images
            generate_images[model_name] = rectified_spet_like_images
            estimate_residuals[model_name] = estimated_residual
            real_residuals[model_name] = target[model_name] - preliminary_predictions
            pre_images[model_name] = preliminary_predictions

        return pre_images, generate_images, estimate_residuals, real_residuals

    def loss_calculate(self,pre_images, generate_images, estimated_residuals, target_images,real_residuals,input_data):
        total_loss = []
        disc_loss = {}
        prenet_loss = {}
        residual_loss = {}
        for i in range(len(self.drf_list)):
            if i == 0:
                advnet_real_output = self.discriminators[self.drf_list[i]](torch.cat((input_data, target_images[self.drf_list[i]]), dim=1))
                advnet_fake_output = self.discriminators[self.drf_list[i]](torch.cat((input_data, generate_images[self.drf_list[i]]), dim=1))
            else:
                advnet_real_output = self.discriminators[self.drf_list[i]](torch.cat((generate_images[self.drf_list[i - 1]], target_images[self.drf_list[i]]), dim=1))
                advnet_fake_output = self.discriminators[self.drf_list[i]](torch.cat((generate_images[self.drf_list[i - 1]], generate_images[self.drf_list[i]]), dim=1))

            adv_real_loss = torch.mean((advnet_real_output - 1) ** 2)
            adv_fake_loss = torch.mean(advnet_fake_output ** 2)
            disc_loss[self.drf_list[i]] = adv_real_loss+adv_fake_loss
            prenet_loss[self.drf_list[i]] = self.lambda_content_arnet * self.content_loss(pre_images[self.drf_list[i]], target_images[self.drf_list[i]])
            residual_loss[self.drf_list[i]] = self.lambda_content_arnet * self.content_loss(real_residuals[self.drf_list[i]],estimated_residuals[self.drf_list[i]])
        for drf in prenet_loss.keys():
            drf_num = drf.split('_')[1]
            if drf_num != 'dose':
                drf_num = int(drf_num)
            else:
                drf_num = 1
            if isinstance(drf_num,int):
                loss = drf_num*(prenet_loss[drf]+residual_loss[drf]+disc_loss[drf])
                total_loss.append(loss)
        print(total_loss)

        # Initialize a variable to store the sum
        total_sum = torch.zeros_like(total_loss[0])  # Initialize with zeros, with the same shape as the first tensor

        # Loop through the list and accumulate the sum
        for tensor in total_loss:
            total_sum += tensor
        return total_sum

    def evaluation_calculate(self,generate,target):
        psnr = self.psnr(generate, target)
        mse = self.mse(generate, target)
        nrmse = NRMSE(generate, target)
        ssim = self.ssim(generate, target)
        return psnr, ssim, mse, nrmse


    def validate(self):
        logger.info('Validating...')
        val_loss = RunningAverage()
        val_psnr = RunningAverage()
        val_ssim = RunningAverage()
        val_mse = RunningAverage()
        val_nrmse = RunningAverage()
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                    enumerate(self.loader['val']), total=len(self.loader['val']),
                    desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration + 1

                val_lpet_images, val_target_images, val_weight = self._split_training_batch(t)
                val_pre_images, val_generate_images, val_estimate_residuals, val_real_residuals = self.forward_pass(val_lpet_images,val_target_images)
                val_loss_iter = self.loss_calculate(val_pre_images, val_generate_images, val_estimate_residuals, val_target_images,val_real_residuals, val_lpet_images)
                val_loss.update(val_loss_iter.item(),self._batch_size(val_lpet_images))
                val_psnr_iter, val_ssim_iter, val_mse_iter, val_nrmse_iter = self.evaluation_calculate(val_generate_images['Full_dose'],val_target_images['Full_dose'])
                val_psnr.update(val_psnr_iter.item(), self._batch_size(val_lpet_images))
                val_ssim.update(val_ssim_iter.item(), self._batch_size(val_lpet_images))
                val_mse.update(val_mse_iter.item(), self._batch_size(val_lpet_images))
                val_nrmse.update(val_nrmse_iter.item(), self._batch_size(val_lpet_images))
            logger.info(f"Validation Loss: {val_loss.avg:.4f}, val_psnr: {val_psnr.avg}, val_mse: {val_mse.avg}, val_nrmse: {val_nrmse.avg}, val_ssim: {val_ssim.avg}")  # Print the average validation loss
            self._log_stats('validation', val_loss.avg, val_loss_iter.item(), val_psnr.avg, val_mse.avg, val_nrmse.avg,val_ssim.avg)
            self._log_images(val_lpet_images, val_target_images['Full_dose'], val_generate_images['Full_dose'],val_pre_images['Full_dose'], val_real_residuals['Full_dose'], val_estimate_residuals['Full_dose'],f'val_')



        return {"val_loss": val_loss.avg,
                "val_psnr": val_psnr.avg,
                "val_mse": val_mse.avg,
                "val_nrmse": val_nrmse.avg,
                "val_ssim": val_ssim.avg}

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def save_joint_checkpoint(self, state, is_best, checkpoint_dir, model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name + '_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name + '_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        state_dict = {}
        refine_state_dict = {}
        dis_state_dict = {}
        for model_name in self.drf_list:
            if isinstance(self.generators[model_name], torch.nn.DataParallel):
                state_dict[model_name] = self.generators[model_name].module.state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].module.state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].module.state_dict()
            else:
                state_dict[model_name] = self.generators[model_name].state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].state_dict()

            last_file_path = os.path.join(self.checkpoint_dir, model_name, 'last_checkpoint.pytorch')
            logger.info(f"Saving checkpoint to '{last_file_path}'")
            # save model state
            self.save_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizer_pre.state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name))
            # save refine model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': refine_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizer_refine.state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='refine')
            # save discriminal model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': dis_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizer_disc.state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='disc')

    def _log_lr(self):
        lr = self.optimizer_pre.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)

    def _log_stats(self, phase, loss_avg, loss, psnr, mse, nrmse, ssim):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss': loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse': mse,
            f'{phase}_nrmse': nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for drf in self.drf_list:
            for name, value in self.generators[drf].named_parameters():
                # print(name,value)
                self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            for name, value in self.refiners[drf].named_parameters():
                # print(name,value)
                self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            for name, value in self.discriminators[drf].named_parameters():
                # print(name,value)
                self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
                # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual': residual,
            'estimated_res': estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        # print(t)
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    input = input.to(self.device)
                if isinstance(input, dict):
                    for name, data in input.items():
                        input[name] = input[name].to(self.device)
                # print(input)
                return input

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight




def create_DRFARGAN_3d_no_residual_trainer_total_back(config):
    drf_list = [ 'drf_20',  'drf_4',  'Full_dose']
    generate_models = {drf: define_G(**config['generator']) for drf in drf_list}
    # generate_models = {drf: get_model(config['generator']) for drf in drf_list}
    # refine_models = {drf: get_model(config['refine_model']) for drf in drf_list}
    discriminate_models = {drf: get_model(config['discriminator']) for drf in drf_list}

    # Get the model we need
    # Get the device
    device = torch.device(config['device'])
    # device_ids = [0,1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        for drf, model in generate_models.items():
            generate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        # for drf, model in refine_models.items():
        #     refine_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

    else:
        for drf, model in generate_models.items():
            generate_models[drf] = model.to(device)

        # for drf, model in refine_models.items():
        #     refine_models[drf] = model.to(device)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = model.to(device)

    logger.info(f"Sending the model to '{config['device']}'")
    # # Put the model on the device
    # generate_model = generate_model.to(device)
    # refine_model = refine_model.to(device)
    # discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()  # MSELoss() #CrossEntropyLoss() #
    criterion_content = L1Loss()
    # Create evaluation metrics
    # psnr = torch.nn.DataParallel(PeakSignalNoiseRatio(), device_ids=device_ids)
    # mse = torch.nn.DataParallel(MeanSquaredError(), device_ids=device_ids)
    # ssim = torch.nn.DataParallel(StructuralSimilarityIndexMeasure(), device_ids=device_ids)
    # torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)
    optimizer_pre = create_optimizer(config['optimizer'], generate_models)
    # optimizer_refine =create_optimizer(config['optimizer'], refine_models)
    optimizer_disc = create_optimizer(config['optimizer'], discriminate_models)

    # optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    # optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    # optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return DRFARGANTrainer_no_residual_total_back(
        drf_list = drf_list,
        generate_models=generate_models,
        # refine_models=refine_models,
        dis_models=discriminate_models,
        optimizer_pre=optimizer_pre,
        #  lr_scheduler_pre=lr_scheduler_pre,
        # optimizer_refine=optimizer_refine,
        #  lr_scheduler_refine=lr_scheduler_refine,
        optimizer_disc=optimizer_disc,
        adv_loss=criterion_adv,
        content_loss=criterion_content,
        psnr=psnr,
        mse=mse,
        ssim=ssim,
        tensorboard_formatter=tensorboard_formatter,
        device=config['device'],
        loaders=loaders,
        resume=resume,
        pre_trained=pre_trained,
        lr=lr,
        **trainer_config)




class DRFARGANTrainer_no_residual_total_back:
    def __init__(self, drf_list,generate_models,  dis_models, optimizer_pre, optimizer_disc,
                 lr, adv_loss, content_loss, psnr,
                 mse, ssim, device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.drf_list = drf_list
        self.generators = generate_models
        # self.refiners = refine_models
        self.discriminators = dis_models
        self.optimizer_pre = optimizer_pre
        # self.scheduler_pre = lr_scheduler_pre
        # self.optimizer_refine = optimizer_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizer_disc = optimizer_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generators, self.optimizer_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generators, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0, self.max_num_epochs)
        for epoch in tqdm.tqdm(
                enumerate(epoch_list), total=self.max_num_epochs,
                desc='Train epoch==%d' % self.num_epochs, ncols=80,
                leave=False):

            lr = self.check_lr(epoch[1], decay_epoch=100)
            for param_group in self.optimizer_pre.param_groups:
                param_group['lr'] = lr
            # for param_group in self.optimizer_refine.param_groups:
            #     param_group['lr'] = lr
            for param_group in self.optimizer_disc.param_groups:
                param_group['lr'] = lr

            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self, epoch, decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse = RunningAverage()
        # train_nrmse = RunningAverage()

        for model_name, gen in self.generators.items():
            gen.train()
            # self.refiners[model_name].train()
            self.discriminators[model_name].train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations, self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images, target_images, weight = self._split_training_batch(t)

            generate_images = self.forward_pass(lpet_images,target_images)

            loss = self.loss_calculate(generate_images,target_images, lpet_images)
            train_loss.update(loss.item(),self._batch_size(lpet_images))

            psnr, ssim, mse, nrmse = self.evaluation_calculate(generate_images['Full_dose'],target_images['Full_dose'])

            self.optimizer_pre.zero_grad()
            # self.optimizer_refine.zero_grad()
            self.optimizer_disc.zero_grad()
            loss.backward()
            self.optimizer_pre.step()
            # self.optimizer_refine.step()
            self.optimizer_disc.step()

            if self.num_iterations % self.validate_after_iters == 0:
                print('start val')
                # set the models in eval mode
                for model_name, gen in self.generators.items():
                    gen.eval()
                    # self.refiners[model_name].eval()
                    self.discriminators[model_name].eval()
                # evaluate on validation set
                val_result = self.validate()
                for model_name, gen in self.generators.items():
                    # set the model back to training mode
                    gen.train()
                    # self.refiners[model_name].train()
                    self.discriminators[model_name].train()

                    # log current learning rate in tensorboard
                    self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss'])
                    # is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')

            if self.num_iterations % self.log_after_iters == 0:
                    # compute eval criterion
                    log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}],Total_Loss: {loss} ," \
                                f"train_psnr: {psnr:.4f}, train_mse: {mse:.4f}, train_nrmse: {nrmse:.4f}, train_ssim: {ssim:.4f}"  #,
                        # log stats, params and images
                    logger.info(log_message)
                    # print('log_stats',drf)
                    self._log_stats(f'train',train_loss.avg, loss.item(),psnr,mse,nrmse,ssim)
                    # print('log_params', drf)
                    self._log_params()
                    # print('log_images', drf)
                    self._log_images(lpet_images, target_images['Full_dose'], generate_images['Full_dose'], f'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1
        return False

    def forward_pass(self,input_data,target):
        input = input_data
        generate_images = {}
        for model_name, gen in self.generators.items():
            # print(model_name)
            # print('gen input',input_data.shape)
            preliminary_predictions = gen(input)
#             print('gen output')
#             rectified_parameters = self.refiners[model_name](preliminary_predictions)
#             print('refine')
#             estimated_residual = preliminary_predictions * rectified_parameters
#             rectified_spet_like_images = preliminary_predictions + estimated_residual
#             print('residual')
            input = preliminary_predictions
            generate_images[model_name] = preliminary_predictions
            # estimate_residuals[model_name] = estimated_residual
            # real_residuals[model_name] = target[model_name] - preliminary_predictions
            # pre_images[model_name] = preliminary_predictions

        return generate_images

    def loss_calculate(self, pre_images,  target_images,input_data):
        total_loss = []
        disc_loss = {}
        prenet_loss = {}
        # residual_loss = {}
        for i in range(len(self.drf_list)):
            if i == 0:
                advnet_real_output = self.discriminators[self.drf_list[i]](torch.cat((input_data, target_images[self.drf_list[i]]), dim=1))
                advnet_fake_output = self.discriminators[self.drf_list[i]](torch.cat((input_data, pre_images[self.drf_list[i]]), dim=1))
            else:
                advnet_real_output = self.discriminators[self.drf_list[i]](torch.cat((pre_images[self.drf_list[i - 1]], target_images[self.drf_list[i]]), dim=1))
                advnet_fake_output = self.discriminators[self.drf_list[i]](torch.cat((pre_images[self.drf_list[i - 1]], pre_images[self.drf_list[i]]), dim=1))

            adv_real_loss = torch.mean((advnet_real_output - 1) ** 2)
            adv_fake_loss = torch.mean(advnet_fake_output ** 2)
            disc_loss[self.drf_list[i]] = adv_real_loss+adv_fake_loss
            prenet_loss[self.drf_list[i]] = self.lambda_content_arnet * self.content_loss(pre_images[self.drf_list[i]], target_images[self.drf_list[i]])
            # residual_loss[self.drf_list[i]] = self.lambda_content_arnet * self.content_loss(real_residuals[self.drf_list[i]],estimated_residuals[self.drf_list[i]])
        for drf in prenet_loss.keys():
            drf_num = drf.split('_')[1]
            if drf_num != 'dose':
                drf_num = int(drf_num)
            else:
                drf_num = 1
            if isinstance(drf_num,int):
                loss = drf_num*(prenet_loss[drf]+disc_loss[drf])
                total_loss.append(loss)
        print(total_loss)

        # Initialize a variable to store the sum
        total_sum = torch.zeros_like(total_loss[0])  # Initialize with zeros, with the same shape as the first tensor

        # Loop through the list and accumulate the sum
        for tensor in total_loss:
            total_sum += tensor
        return total_sum

    def evaluation_calculate(self,generate,target):
        psnr = self.psnr(generate, target)
        mse = self.mse(generate, target)
        nrmse = NRMSE(generate, target)
        ssim = self.ssim(generate, target)
        return psnr, ssim, mse, nrmse


    def validate(self):
        logger.info('Validating...')
        val_loss = RunningAverage()
        val_psnr = RunningAverage()
        val_ssim = RunningAverage()
        val_mse = RunningAverage()
        val_nrmse = RunningAverage()
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                    enumerate(self.loader['val']), total=len(self.loader['val']),
                    desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration + 1

                val_lpet_images, val_target_images, val_weight = self._split_training_batch(t)
                val_generate_images = self.forward_pass(val_lpet_images,val_target_images)
                val_loss_iter = self.loss_calculate( val_generate_images, val_target_images, val_lpet_images)
                val_loss.update(val_loss_iter.item(),self._batch_size(val_lpet_images))
                val_psnr_iter, val_ssim_iter, val_mse_iter, val_nrmse_iter = self.evaluation_calculate(val_generate_images['Full_dose'],val_target_images['Full_dose'])
                val_psnr.update(val_psnr_iter.item(), self._batch_size(val_lpet_images))
                val_ssim.update(val_ssim_iter.item(), self._batch_size(val_lpet_images))
                val_mse.update(val_mse_iter.item(), self._batch_size(val_lpet_images))
                val_nrmse.update(val_nrmse_iter.item(), self._batch_size(val_lpet_images))
            logger.info(f"Validation Loss: {val_loss.avg:.4f}, val_psnr: {val_psnr.avg}, val_mse: {val_mse.avg}, val_nrmse: {val_nrmse.avg}, val_ssim: {val_ssim.avg}")  # Print the average validation loss
            self._log_stats('validation', val_loss.avg, val_loss_iter.item(), val_psnr.avg, val_mse.avg, val_nrmse.avg,val_ssim.avg)
            self._log_images(val_lpet_images, val_target_images['Full_dose'], val_generate_images['Full_dose'],f'val_')



        return {"val_loss": val_loss.avg,
                "val_psnr": val_psnr.avg,
                "val_mse": val_mse.avg,
                "val_nrmse": val_nrmse.avg,
                "val_ssim": val_ssim.avg}

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def save_joint_checkpoint(self, state, is_best, checkpoint_dir, model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name + '_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name + '_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        state_dict = {}
        refine_state_dict = {}
        dis_state_dict = {}
        for model_name in self.drf_list:
            if isinstance(self.generators[model_name], torch.nn.DataParallel):
                state_dict[model_name] = self.generators[model_name].module.state_dict()
                # refine_state_dict[model_name] = self.refiners[model_name].module.state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].module.state_dict()
            else:
                state_dict[model_name] = self.generators[model_name].state_dict()
                # refine_state_dict[model_name] = self.refiners[model_name].state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].state_dict()

            last_file_path = os.path.join(self.checkpoint_dir, model_name, 'last_checkpoint.pytorch')
            logger.info(f"Saving checkpoint to '{last_file_path}'")
            # save model state
            self.save_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizer_pre.state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name))
            # save refine model state
            # self.save_joint_checkpoint({
            #     'num_epochs': self.num_epochs + 1,
            #     'num_iterations': self.num_iterations,
            #     'model_state_dict': refine_state_dict[model_name],
            #     'best_eval_score': self.best_eval_score,
            #     'optimizer_state_dict': self.optimizer_refine.state_dict(),
            # }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='refine')
            # save discriminal model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': dis_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizer_disc.state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='disc')

    def _log_lr(self):
        lr = self.optimizer_pre.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)

    def _log_stats(self, phase, loss_avg, loss, psnr, mse, nrmse, ssim):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss': loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse': mse,
            f'{phase}_nrmse': nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for drf in self.drf_list:
            for name, value in self.generators[drf].named_parameters():
                # print(name,value)
                self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # for name, value in self.refiners[drf].named_parameters():
            #     # print(name,value)
            #     self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            for name, value in self.discriminators[drf].named_parameters():
                # print(name,value)
                self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
                # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        # print(t)
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    input = input.to(self.device)
                if isinstance(input, dict):
                    for name, data in input.items():
                        input[name] = input[name].to(self.device)
                # print(input)
                return input

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight


def create_DRFARGAN_3d_trainer_no_residual(config):
    drf_list = ['drf_20', 'drf_4', 'Full_dose']
    generate_models = {drf: define_G(**config['generator']) for drf in drf_list}
    refine_models = {drf: get_model(config['refine_model']) for drf in drf_list}
    discriminate_models = {drf: get_model(config['discriminator']) for drf in drf_list}

    # Get the model we need
    # Get the device
    device = torch.device(config['device'])
    # device_ids = [0,1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        for drf, model in generate_models.items():
            generate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in refine_models.items():
            refine_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = nn.DataParallel(model).to(device)  # , device_ids=device_ids)

    else:
        for drf, model in generate_models.items():
            generate_models[drf] = model.to(device)

        for drf, model in refine_models.items():
            refine_models[drf] = model.to(device)

        for drf, model in discriminate_models.items():
            discriminate_models[drf] = model.to(device)

    # generate_model = torch.nn.DataParallel(generate_model, device_ids=device_ids)
    # refine_model = torch.nn.DataParallel(refine_model, device_ids=device_ids)
    # discriminate_model = torch.nn.DataParallel(discriminate_model, device_ids=device_ids)

    logger.info(f"Sending the model to '{config['device']}'")
    # # Put the model on the device
    # generate_model = generate_model.to(device)
    # refine_model = refine_model.to(device)
    # discriminate_model = discriminate_model.to(device)
    # Get loss function
    # Get loss function
    criterion_adv = BCELoss()  # MSELoss() #CrossEntropyLoss() #
    criterion_content = L1Loss()
    # Create evaluation metrics
    # psnr = torch.nn.DataParallel(PeakSignalNoiseRatio(), device_ids=device_ids)
    # mse = torch.nn.DataParallel(MeanSquaredError(), device_ids=device_ids)
    # ssim = torch.nn.DataParallel(StructuralSimilarityIndexMeasure(), device_ids=device_ids)
    # torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    psnr = PeakSignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    # Create data loaders
    # print('config',config)
    loaders = get_train_loaders(config)
    optimizers_pre = {drf: create_optimizer(config['optimizer'], generate_models[drf]) for drf in drf_list}
    optimizers_refine = {drf: create_optimizer(config['optimizer'], refine_models[drf]) for drf in drf_list}
    optimizers_disc = {drf: create_optimizer(config['optimizer'], discriminate_models[drf]) for drf in drf_list}

    # optimizer_pre = create_optimizer(config['optimizer'], generate_model)
    # optimizer_refine = create_optimizer(config['optimizer'], refine_model)
    # optimizer_disc = create_optimizer(config['optimizer'], discriminate_model)
    lr = config['optimizer']['learning_rate']
    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    logger.info(f'the learning rate config of learning rate schedule is {lr_config}')
    # lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    # lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)
    print('drf_list', drf_list)

    return DRFARGANTrainer(
        generate_models=generate_models,
        refine_models=refine_models,
        dis_models=discriminate_models,
        optimizers_pre=optimizers_pre,
        #  lr_scheduler_pre=lr_scheduler_pre,
        optimizers_refine=optimizers_refine,
        #  lr_scheduler_refine=lr_scheduler_refine,
        optimizers_disc=optimizers_disc,
        adv_loss=criterion_adv,
        content_loss=criterion_content,
        psnr=psnr,
        mse=mse,
        ssim=ssim,
        tensorboard_formatter=tensorboard_formatter,
        device=config['device'],
        loaders=loaders,
        resume=resume,
        pre_trained=pre_trained,
        lr=lr,
        drf_list=drf_list,
        **trainer_config)


class DRFARGANTrainer_np_residual:
    def __init__(self, generate_models, refine_models, dis_models, optimizers_pre, optimizers_disc, optimizers_refine,
                 lr, drf_list, adv_loss, content_loss, psnr, mse, ssim, device, loaders, checkpoint_dir, max_num_epochs,
                 max_num_iterations,
                 validate_after_iters=1000, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):
        self.generators = generate_models
        self.refiners = refine_models
        self.discriminators = dis_models
        self.optimizers_pre = optimizers_pre
        # self.scheduler_pre = lr_scheduler_pre
        self.optimizers_refine = optimizers_refine
        # self.scheduler_refine = lr_scheduler_refine
        self.optimizers_disc = optimizers_disc
        self.adv_loss = adv_loss
        self.content_loss = content_loss
        self.drf_list = drf_list
        self.psnr = psnr
        self.mse = mse
        self.ssim = ssim
        self.device = device
        self.loader = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.lr = lr
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.lambda_content_arnet = kwargs['lambda_content_prenet']
        self.lambda_content_prenet = kwargs['lambda_content_arnet']

        # logger.info(generate_model)
        # logger.info(refine_model)
        # logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')
        logger.info(f"check_dir:{checkpoint_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        logger.info("finish the summarywriter")

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.generators, self.optimizers_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.generators, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0, self.max_num_epochs)
        for epoch in tqdm.tqdm(
                enumerate(epoch_list), total=self.max_num_epochs,
                desc='Train epoch==%d' % self.num_epochs, ncols=80,
                leave=False):

            lr = self.check_lr(epoch[1], decay_epoch=100)
            for drf in self.drf_list:
                for param_group in self.optimizers_pre[drf].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizers_refine[drf].param_groups:
                    param_group['lr'] = lr
                for param_group in self.optimizers_disc[drf].param_groups:
                    param_group['lr'] = lr

            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def check_lr(self, epoch, decay_epoch):
        num_epochs = self.max_num_epochs
        learning_rate = self.lr
        # print(epoch,decay_epoch)
        epoch = int(epoch)
        decay_epoch = int(decay_epoch)
        if epoch < decay_epoch:
            current_lr = learning_rate
        else:
            current_lr = learning_rate * (1 - (epoch - decay_epoch) / (num_epochs - decay_epoch))
        return current_lr

    def train(self):
        train_loss = RunningAverage()
        # train_psnr = RunningAverage()
        # train_ssim = RunningAverage()
        # train_mse = RunningAverage()
        # train_nrmse = RunningAverage()

        for model_name, gen in self.generators.items():
            gen.train()
            self.refiners[model_name].train()
            self.discriminators[model_name].train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loader['train']), total=len(self.loader['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations, self.num_epochs), ncols=80, leave=False):
            # print(t[0].shape,t[1].shape)
            lpet_images, target_images, weight = self._split_training_batch(t)
            generate_images = {}

            input_data = lpet_images
            input_images = {}
            total_loss = {}
            disc_loss = {}
            prenet_loss = {}
            residual_loss = {}
            psnr_drf = {}
            ssim_drf = {}
            mse_drf = {}
            nrmse_drf = {}
            pre_predictions = {}
            real_residuals = {}
            estimated_residuals = {}
            all_model_loss = 0
            for model_name, gen in self.generators.items():
                # print(model_name)
                input_images[model_name] = input_data
                # print('gen input',input_data.shape)
                preliminary_predictions = gen(input_data)
                #                 print('gen output',preliminary_predictions.shape)
                rectified_parameters = self.refiners[model_name](preliminary_predictions)
                estimated_residuals[model_name] = preliminary_predictions * rectified_parameters
                rectified_spet_like_images = preliminary_predictions + estimated_residuals[model_name]
                pre_predictions[model_name] = preliminary_predictions
                #                 print('residual')
                generate_images[model_name] = rectified_spet_like_images
                real_residuals[model_name] = target_images[model_name] - preliminary_predictions

                #                 print('start loss')
                advnet_real_output = self.discriminators[model_name](
                    torch.cat((input_data, target_images[model_name]), dim=1).detach())
                advnet_fake_output = self.discriminators[model_name](
                    torch.cat((input_data, generate_images[model_name]), dim=1).detach())  # (fake_images.detach())

                adv_real_loss = torch.mean((advnet_real_output - 1) ** 2)
                adv_fake_loss = torch.mean(advnet_fake_output ** 2)

                disc_loss[model_name] = adv_real_loss + adv_fake_loss

                # AR-Net: Train AR-Net with content and adversarial losses
                prenet_loss[model_name] = self.lambda_content_prenet * self.content_loss(preliminary_predictions,
                                                                                         target_images[model_name])
                # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
                # print(f'content_loss_prenet{content_loss_prenet}')

                residual_loss[model_name] = self.lambda_content_arnet * self.content_loss(real_residuals[model_name],
                                                                                          estimated_residuals[
                                                                                              model_name])

                total_loss[model_name] = disc_loss[model_name] + prenet_loss[model_name] + residual_loss[model_name]
                # total_loss[model_name] = arnet_loss
                all_model_loss += total_loss[model_name]
                # print(total_loss)
                # print(arnet_loss)
                # print(all_model_loss)

                # print(f'arnet_loss{arnet_loss}')
                psnr = self.psnr(generate_images[model_name], target_images[model_name])
                psnr_drf[model_name] = psnr
                # train_psnr.update(psnr,self._batch_size(lpet_images))
                mse = self.mse(generate_images[model_name], target_images[model_name])
                mse_drf[model_name] = mse
                #             train_mse.update(mse,self._batch_size(lpet_images))
                nrmse = NRMSE(generate_images[model_name], target_images[model_name])
                nrmse_drf[model_name] = nrmse
                #             train_nrmse.update(nrmse,self._batch_size(lpet_images))
                ssim = self.ssim(generate_images[model_name], target_images[model_name])
                ssim_drf[model_name] = ssim
                #             train_ssim.update(ssim,self._batch_size(lpet_images))
                input_data = generate_images[model_name].detach()
                # train_loss.update(all_model_loss, self._batch_size(input_data))

                # for model_name, optimize_pre in self.optimizers_pre.items():
                self.optimizers_pre[model_name].zero_grad()
                self.optimizers_refine[model_name].zero_grad()
                self.optimizers_disc[model_name].zero_grad()
                # print('finish zero grad')
                # all_model_loss.backward(retain_graph=True)
                total_loss[model_name].backward()
                # print('finish backward')
                self.optimizers_pre[model_name].step()
                self.optimizers_refine[model_name].step()
                self.optimizers_disc[model_name].step()
            train_loss.update(all_model_loss, self._batch_size(input_data))

            if self.num_iterations % self.validate_after_iters == 0:
                print('start val')
                # set the models in eval mode
                for model_name, gen in self.generators.items():
                    gen.eval()
                    self.refiners[model_name].eval()
                    self.discriminators[model_name].eval()
                # evaluate on validation set
                val_result = self.validate()
                for model_name, gen in self.generators.items():
                    # set the model back to training mode
                    gen.train()
                    self.refiners[model_name].train()
                    self.discriminators[model_name].train()

                    # # adjust learning rate if necessary
                    # if isinstance(self.scheduler, ReduceLROnPlateau):
                    #     self.scheduler.step(eval_score)
                    # else:
                    #     self.scheduler.step()

                    # log current learning rate in tensorboard
                    self._log_lr(model_name)
                # remember best validation metric
                is_best = self._is_best_eval_score(val_result['val_loss']['Full_dose'])
                # is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                logger.info('start save check_point')
                self._save_checkpoint(is_best)
                logger.info('finish save check_point')

            if self.num_iterations % self.log_after_iters == 0:
                for drf in self.drf_list:
                    # compute eval criterion
                    log_message = f"Batch [{batch_idx + 1}/{len(self.loader['train'])}],Total_Loss: {all_model_loss} , {drf} as the input: " \
                                  f"Loss: {total_loss[drf].item():.4f}, content_loss_prenet: {prenet_loss[drf].item():.4f}," \
                                  f"content_loss_arnet: {residual_loss[drf].item():.4f}, adv_loss: {disc_loss[drf].item():.4f}," \
                                  f"train_psnr: {psnr_drf[drf]:.4f}, train_mse: {mse_drf[drf]:.4f}, train_nrmse: {nrmse_drf[drf]:.4f}, train_ssim: {ssim_drf[drf]:.4f}"  # ,
                    # log stats, params and images
                    logger.info(log_message)
                    # print('log_stats',drf)
                    self._log_stats(f'train_{drf}', all_model_loss, total_loss[drf].item(), prenet_loss[drf].item(),
                                    residual_loss[drf].item(), disc_loss[drf].item(), psnr_drf[drf], mse_drf[drf],
                                    nrmse_drf[drf], ssim_drf[drf])
                    # print('log_params', drf)
                    self._log_params(drf)
                    # print('log_images', drf)
                    self._log_images(input_images[drf], target_images[drf], generate_images[drf], pre_predictions[drf],
                                     real_residuals[drf], estimated_residuals[drf], f'train_{drf}_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def validate(self):
        logger.info('Validating...')
        val_loss = {}
        val_disc_loss = {}
        val_prenet_loss = {}
        val_residual_loss = {}
        val_psnr_dict = {}
        val_ssim_dict = {}
        val_mse_dict = {}
        val_nrmse_dict = {}

        for drf in self.drf_list:
            val_loss[drf] = 0
            val_prenet_loss[drf] = 0
            val_residual_loss[drf] = 0
            val_disc_loss[drf] = 0
            val_psnr_dict[drf] = 0
            val_ssim_dict[drf] = 0
            val_mse_dict[drf] = 0
            val_nrmse_dict[drf] = 0
        val_loss_all_model = 0
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                    enumerate(self.loader['val']), total=len(self.loader['val']),
                    desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):

                val_iteration = val_iteration + 1

                val_lpet_images, val_target_images, val_weight = self._split_training_batch(t)
                val_generate_images = {}
                val_input_data = val_lpet_images
                val_total_loss = {}
                val_input_images = {}
                val_pre_predictions = {}
                val_real_residuals = {}
                val_estimated_residuals = {}
                val_all_model_loss = 0
                for model_name, val_gen in self.generators.items():
                    print('val', model_name)
                    val_input_images[model_name] = val_input_data
                    val_preliminary_predictions = val_gen(val_input_data)
                    val_rectified_parameters = self.refiners[model_name](val_preliminary_predictions)
                    val_estimated_residual = val_preliminary_predictions * val_rectified_parameters
                    val_rectified_spet_like_images = val_preliminary_predictions + val_estimated_residual
                    val_pre_predictions[model_name] = val_preliminary_predictions

                    val_estimated_residuals[model_name] = val_estimated_residual
                    val_target = val_target_images[model_name]
                    val_generate_images[model_name] = val_rectified_spet_like_images

                    val_advnet_real_output = self.discriminators[model_name](
                        torch.cat((val_input_data, val_target), dim=1))
                    val_advnet_fake_output = self.discriminators[model_name](
                        torch.cat((val_input_data, val_generate_images[model_name]), dim=1))  # (fake_images.detach())

                    val_adv_real_loss = torch.mean((val_advnet_real_output - 1) ** 2)
                    val_adv_fake_loss = torch.mean(val_advnet_fake_output ** 2)

                    val_adv_loss = val_adv_real_loss + val_adv_fake_loss
                    val_disc_loss[model_name] += val_adv_loss
                    # AR-Net: Train AR-Net with content and adversarial losses
                    val_content_loss_prenet = self.lambda_content_prenet * self.content_loss(
                        val_preliminary_predictions, val_target)
                    val_prenet_loss[model_name] += val_content_loss_prenet
                    # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
                    # print(f'content_loss_prenet{content_loss_prenet}')

                    val_real_residual = val_target - val_preliminary_predictions
                    val_real_residuals[model_name] = val_real_residual
                    val_content_loss_arnet = self.lambda_content_arnet * self.content_loss(val_real_residual,
                                                                                           val_estimated_residual)
                    val_residual_loss[model_name] += val_content_loss_arnet

                    val_arnet_loss = val_adv_loss + val_content_loss_prenet + val_content_loss_arnet
                    val_all_model_loss += val_arnet_loss

                    val_loss[model_name] += val_arnet_loss

                    # print(f'arnet_loss{arnet_loss}')
                    val_psnr = self.psnr(val_generate_images[model_name], val_target)
                    val_psnr_dict[model_name] += val_psnr
                    val_mse = self.mse(val_generate_images[model_name], val_target)
                    val_mse_dict[model_name] += val_mse
                    val_nrmse = NRMSE(val_generate_images[model_name], val_target)
                    val_nrmse_dict[model_name] += val_nrmse
                    val_ssim = self.ssim(val_generate_images[model_name], val_target)
                    val_ssim_dict[model_name] += val_ssim
                    val_input_data = val_generate_images[model_name].detach()
                val_loss_all_model += val_all_model_loss
            num = len(self.loader['val'])
            dicts_list = [val_loss, val_disc_loss, val_prenet_loss, val_residual_loss, val_psnr_dict, val_ssim_dict,
                          val_mse_dict, val_nrmse_dict]
            for my_dict in dicts_list:
                for key in my_dict:
                    my_dict[key] = my_dict[key] / num

            for drf in self.drf_list:
                num = len(self.loader['val'])
                # compute eval criterion
                log_message = f"Total_Loss: {val_loss_all_model / num} , {drf} as the target: " \
                              f"Loss: {val_loss[drf]:.4f}, content_loss_prenet: {val_prenet_loss[drf]:.4f}," \
                              f"content_loss_arnet: {val_residual_loss[drf]:.4f}, adv_loss: {val_disc_loss[drf]:.4f}," \
                              f"val_psnr: {val_psnr_dict[drf]:.4f}, val_mse: {val_mse_dict[drf]:.4f}, val_nrmse: {val_nrmse_dict[drf]:.4f}, val_ssim: {val_ssim_dict[drf]:.4f}"  # ,
                # log stats, params and images
                logger.info(log_message)
                self._log_stats(f'val_{drf}', val_loss_all_model / num, val_loss[drf], val_prenet_loss[drf],
                                val_residual_loss[drf], val_disc_loss[drf], val_psnr_dict[drf], val_mse_dict[drf],
                                val_nrmse_dict[drf], val_ssim_dict[drf])
                self._log_images(val_input_images[drf], val_target_images[self.drf_list[self.drf_list.index(drf)]],
                                 val_generate_images[drf], val_pre_predictions[drf], val_real_residuals[drf],
                                 val_estimated_residuals[drf], f'val_{drf}_')

        return {"val_all_model_loss": val_loss_all_model / num, "val_loss": val_loss, "val_psnr": val_psnr_dict,
                "val_mse": val_mse_dict, "val_nrmse": val_nrmse_dict, "val_ssim": val_ssim_dict}

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def save_checkpoint(self, state, is_best, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def save_joint_checkpoint(self, state, is_best, checkpoint_dir, model_name):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        last_file_path = os.path.join(checkpoint_dir, model_name + '_last_checkpoint.pytorch')
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, model_name + '_best_checkpoint.pytorch')
            shutil.copyfile(last_file_path, best_file_path)

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        state_dict = {}
        refine_state_dict = {}
        dis_state_dict = {}
        for model_name, gen in self.generators.items():
            if isinstance(self.generators[model_name], torch.nn.DataParallel):
                state_dict[model_name] = self.generators[model_name].module.state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].module.state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].module.state_dict()
            else:
                state_dict[model_name] = self.generators[model_name].state_dict()
                refine_state_dict[model_name] = self.refiners[model_name].state_dict()
                dis_state_dict[model_name] = self.discriminators[model_name].state_dict()

            last_file_path = os.path.join(self.checkpoint_dir, model_name, 'last_checkpoint.pytorch')
            logger.info(f"Saving checkpoint to '{last_file_path}'")
            # save model state
            self.save_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_pre[model_name].state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name))
            # save refine model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': refine_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_refine[model_name].state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='refine')
            # save discriminal model state
            self.save_joint_checkpoint({
                'num_epochs': self.num_epochs + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': dis_state_dict[model_name],
                'best_eval_score': self.best_eval_score,
                'optimizer_state_dict': self.optimizers_disc[model_name].state_dict(),
            }, is_best, checkpoint_dir=os.path.join(self.checkpoint_dir, model_name), model_name='disc')

    def _log_lr(self, model_name):
        lr = self.optimizers_pre[model_name].param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)

    def _log_stats(self, phase, all_model_loss, loss, pre_loss, res_loss, disc_loss, psnr, mse, nrmse, ssim):
        tag_value = {
            f'{phase}_loss_all_model': all_model_loss,
            f'{phase}_loss': loss,
            f'{phase}_disc_loss': disc_loss,
            f'{phase}_pre_loss': pre_loss,
            f'{phase}_res_loss': res_loss,
            f'{phase}_psnr': psnr,
            f'{phase}_mse': mse,
            f'{phase}_nrmse': nrmse,
            f'{phase}_ssim': ssim
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self, drf):
        logger.info('Logging model parameters and gradients')
        for name, value in self.generators[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        for name, value in self.refiners[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        for name, value in self.discriminators[drf].named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual': residual,
            'estimated_res': estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

    def _split_training_batch(self, t):
        # print(t)
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    input = input.to(self.device)
                if isinstance(input, dict):
                    for name, data in input.items():
                        input[name] = input[name].to(self.device)
                # print(input)
                return input

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight



