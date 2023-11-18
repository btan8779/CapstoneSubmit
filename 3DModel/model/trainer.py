from Config_doc.logger import get_logger
from model.Model import get_model,define_G
import shutil
import torch
import importlib
from torch import optim,nn
from torch.nn import BCELoss, L1Loss,MSELoss
# from model.losses import SpectralLoss
# from model.losses_v2 import SpectralLoss
from model.losses_v4 import SpectralLoss
from data.dataloader import get_train_loaders
from skimage.metrics import normalized_root_mse
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MeanSquaredError
import copy
from model.tensorboard import DefaultTensorboardFormatter
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm

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

def create_ARGAN_3d_trainer_with_spectrum(config):
    # Get the model we need
    # generate_model = define_G(config['generator'])
    generate_model = get_model(config['generator'])
    refine_model = get_model(config['refine_model'])
    discriminate_model = get_model(config['discriminator'])
    # Get the device
    device = torch.device(config['device'])
    logger.info(f"Sending the model to '{config['device']}'")
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
    # criterion_spectrum = SpectralLoss()
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
    # lambda_specturm = trainer_config['lambda_specturm']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return ARGANTrainer_with_spectrum(
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
                         # spectrum_loss = criterion_spectrum,
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



class ARGANTrainer_with_spectrum:
    def __init__(self, generate_model, refine_model,dis_model, optimizer_pre,optimizer_disc,optimizer_refine,
                 lr, adv_loss, content_loss,psnr,
                 mse,ssim,device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
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
        # self.spectrum_loss = spectrum_loss
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
        self.lambda_specturm = kwargs['lambda_spectrum']

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
            # lpet_images = t['low_dose'].to(self.device)
            # spet_images = t['full_dose'].to(self.device)
            # print(lpet_images.shape,spet_images.shape)

            # PreNet: Generate preliminary predictions
            preliminary_predictions = self.generate(lpet_images)
            # AR-Net: Generate rectified parameters and estimated residual
            rectified_parameters = self.refine(preliminary_predictions)
            estimated_residual = preliminary_predictions * rectified_parameters

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

            # spectral_ai_rectified = SpectralLoss(fake_images)
            # spectral_ai_real = SpectralLoss(real_images)
            # print(f'spectral_loss_rectified: {spectral_ai_rectified}; spectral_loss_real: {spectral_ai_real}')

            # spectral_loss_total = self.lambda_specturm*self.adv_loss(spectral_ai_real,spectral_ai_rectified)
            spectral_loss_total = torch.tensor([0]).to(self.device)
            # AR-Net: Train AR-Net with content and adversarial losses
            content_loss_prenet = self.lambda_content_prenet * self.content_loss(preliminary_predictions, spet_images)
            # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
            # print(f'content_loss_prenet{content_loss_prenet}')

            real_residual = spet_images - preliminary_predictions
            content_loss_arnet = self.lambda_content_arnet * self.content_loss(real_residual, estimated_residual)
            # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )
            # print(f'content_loss_arnet{content_loss_arnet}')
            # Apply spectral regularization loss to AR-Net
            # spectral_loss_rectified = spectral_reg_loss(rectified_spet_like_images)
            # spectral_loss_real = spectral_reg_loss(real_images)

            arnet_loss =adv_loss + content_loss_prenet + content_loss_arnet + spectral_loss_total
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
                          f"Loss: {arnet_loss.item():.4f},spectral_loss: {spectral_loss_total.item():.4f}, content_loss_prenet: {content_loss_prenet.item():.4f},"\
                          f"content_loss_arnet: {content_loss_arnet.item():.4f}, adv_loss: {adv_loss.item():.4f},"\
                          f"train_psnr: {psnr:.4f}, train_mse: {mse:.4f}, train_nrmse: {nrmse:.4f}, train_ssim: {ssim:.4f}"  #,
                # Epoch [{int(epoch) + 1}/{self.num_epochs}],
                # print(log_message)
                # with open(log_file, 'a') as f:
                #     f.write(log_message + '\n')

                # log stats, params and images
                logger.info(log_message)
                self._log_stats('train', train_loss.avg,arnet_loss.item(),content_loss_prenet.item(),content_loss_arnet.item(),adv_loss.item(),spectral_loss_total.item(),psnr,mse,nrmse,ssim)
                self._log_params()
                self._log_images(lpet_images, spet_images, fake_images, preliminary_predictions, real_residual,  estimated_residual, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False
    def validate(self):
        logger.info('Validating...')
        val_loss = RunningAverage()
        val_loss_spectrum = RunningAverage()
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
                val_rectified_parameters = self.refine(val_preliminary_predictions)
                val_estimated_residual = val_preliminary_predictions * val_rectified_parameters
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

                # val_spectral_ai_rectified = SpectralLoss(val_fake_images)
                # val_spectral_ai_real = SpectralLoss(val_real_images)


                # val_spectral_loss_total = self.lambda_specturm*self.adv_loss(val_spectral_ai_real, val_spectral_ai_rectified)
                val_spectral_loss_total = torch.tensor([0]).to(self.device)
                val_loss_spectrum.update(val_spectral_loss_total, self._batch_size(val_lpet_images))

                  # advnet_loss_real = criterion_adv(spectral_loss_real, real_labels)
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

                  # Apply spectral regularization loss to AR-Net
                  # spectral_loss_rectified = spectral_reg_loss(rectified_spet_like_images)
                  # spectral_loss_real = spectral_reg_loss(real_images)
                val_arnet_loss = val_adv_loss + val_content_loss_prenet + val_content_loss_arnet + val_spectral_loss_total
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
                            val_loss_ar.avg,val_loss_adv.avg,val_loss_spectrum.avg, val_psnr.avg, val_mse.avg, val_nrmse.avg, val_ssim.avg)
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
    def _log_stats(self, phase, loss_avg, loss,pre_loss,res_loss, disc_loss,spectrum_loss, psnr,mse,nrmse,ssim):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_loss': loss,
            f'{phase}_disc_loss': disc_loss,
            f'{phase}_pre_loss': pre_loss,
            f'{phase}_res_loss': res_loss,
            f'{phase}_spectrum_loss': spectrum_loss,
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


def create_ARGAN_3d_trainer_without_spectrum(config):
    # Get the model we need
    # generate_model = define_G(config['generator'])
    generate_model = get_model(config['generator'])
    refine_model = get_model(config['refine_model'])
    discriminate_model = get_model(config['discriminator'])
    # Get the device
    device = torch.device(config['device'])
    logger.info(f"Sending the model to '{config['device']}'")
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
    # criterion_spectrum = SpectralLoss()
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
    # lambda_specturm = trainer_config['lambda_specturm']

    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return ARGANTrainer_with_spectrum(
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
                         # spectrum_loss = criterion_spectrum,
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



class ARGANTrainer_without_spectrum:
    def __init__(self, generate_model, refine_model,dis_model, optimizer_pre,optimizer_disc,optimizer_refine,
                 lr, adv_loss, content_loss,psnr,
                 mse,ssim,device, loaders, checkpoint_dir, max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
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
        # self.spectrum_loss = spectrum_loss
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
        self.lambda_specturm = kwargs['lambda_spectrum']

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
            # lpet_images = t['low_dose'].to(self.device)
            # spet_images = t['full_dose'].to(self.device)
            # print(lpet_images.shape,spet_images.shape)

            # PreNet: Generate preliminary predictions
            preliminary_predictions = self.generate(lpet_images)
            # AR-Net: Generate rectified parameters and estimated residual
            estimated_residual = self.refine(preliminary_predictions)

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

            # spectral_ai_rectified = SpectralLoss(fake_images)
            # spectral_ai_real = SpectralLoss(real_images)
            # print(f'spectral_loss_rectified: {spectral_ai_rectified}; spectral_loss_real: {spectral_ai_real}')

            # spectral_loss_total = self.lambda_specturm*self.adv_loss(spectral_ai_real,spectral_ai_rectified)

            # AR-Net: Train AR-Net with content and adversarial losses
            content_loss_prenet = self.lambda_content_prenet * self.content_loss(preliminary_predictions, spet_images)
            # (x = LPET; y = SPET; P = PreNet(),L1(y, P(x)))
            # print(f'content_loss_prenet{content_loss_prenet}')

            real_residual = spet_images - preliminary_predictions
            content_loss_arnet = self.lambda_content_arnet * self.content_loss(real_residual, estimated_residual)
            # (r = SPET - P(x); R = ARNet(P(x)); R(x) = R*P(x), L1(r, R(x)) )
            # print(f'content_loss_arnet{content_loss_arnet}')
            # Apply spectral regularization loss to AR-Net
            # spectral_loss_rectified = spectral_reg_loss(rectified_spet_like_images)
            # spectral_loss_real = spectral_reg_loss(real_images)

            arnet_loss =adv_loss + content_loss_prenet + content_loss_arnet #+ spectral_loss_total
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
        # val_loss_spectrum = RunningAverage()
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
                val_estimated_residual= self.refine(val_preliminary_predictions)
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

                # val_spectral_ai_rectified = SpectralLoss(val_fake_images)
                # val_spectral_ai_real = SpectralLoss(val_real_images)


                # val_spectral_loss_total = self.lambda_specturm*self.adv_loss(val_spectral_ai_real, val_spectral_ai_rectified)
                # val_spectral_loss_total = torch.tensor([0]).to(self.device)
                # val_loss_spectrum.update(val_spectral_loss_total, self._batch_size(val_lpet_images))

                  # advnet_loss_real = criterion_adv(spectral_loss_real, real_labels)
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

                  # Apply spectral regularization loss to AR-Net
                  # spectral_loss_rectified = spectral_reg_loss(rectified_spet_like_images)
                  # spectral_loss_real = spectral_reg_loss(real_images)
                val_arnet_loss = val_adv_loss + val_content_loss_prenet + val_content_loss_arnet #+ val_spectral_loss_total
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












