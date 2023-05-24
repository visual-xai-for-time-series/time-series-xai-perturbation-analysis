import torch

from tqdm import tqdm

from captum.attr import DeepLiftShap, GradientShap, KernelShap, IntegratedGradients, ShapleyValueSampling, Saliency, DeepLift, InputXGradient, Occlusion, LRP


def generate_attributions_batch(shape, model, device, dataloader, baselines):
    attribution_techniques_batch = [
        ['DeepLiftShap', DeepLiftShap, {'baselines': baselines}],
        ['GradientShap', GradientShap, {'baselines': baselines}],
        ['IntegratedGradients', IntegratedGradients, {}],
        ['Saliency', Saliency, {}],
        ['DeepLift', DeepLift, {}],
        ['Occlusion', Occlusion, {'sliding_window_shapes': (1, 5)}],
    #     ['LRP', LRP, {}],
    ]
    
    attributions_train = {}
    predictions_train = {}
    
    model.eval()
    
    for at in attribution_techniques_batch:
        at_name, at_function, kwargs = at
        attribute_tec = at_function(model)

        preds = []
        attributions = []
        for x in tqdm(dataloader, desc=f'Start with {at_name:<25s}'):
            input_, label_ = x
            input_ = input_.reshape(-1, *shape)
            input_ = input_.float().to(device)
            label_ = label_.float().to(device)

            pred = model(input_)
            preds.extend(pred)

            attribution = attribute_tec.attribute(input_, target=torch.argmax(label_, axis=1), **kwargs)
            attributions.extend(attribution)

            del attribution

        attributions = torch.stack(attributions)
        attributions_train[at_name] = attributions.detach().cpu().reshape(-1, shape[-1]).numpy()

        del attributions

        preds = torch.stack(preds)
        predictions_train[at_name] = preds.cpu().detach().numpy()

        del preds

        del attribute_tec
        torch.cuda.empty_cache()
        
    return attributions_train, predictions_train


def generate_attributions_single(shape, model, device, dataloader, baselines):
    attribution_techniques_single = [
        ['KernelShap', KernelShap, {}],
    ]
    
    attributions_train = {}
    predictions_train = {}
    
    model.eval()
    
    for at in attribution_techniques_single:
        at_name, at_function, kwargs = at

        preds = []
        attributions = []
        for x in tqdm(dataloader, desc=f'Start with {at_name:<25s}'):
            input_, label_ = x
            input_ = input_.reshape(input_.shape[0], 1, -1)
            input_ = input_.float().to(device)
            label_ = label_.float().to(device)

            pred = model(input_.reshape(-1, *shape).float().to(device))
            preds.extend(pred)

            attribute_tec = at_function(model)
            for y in range(len(input_)):
                input_temp = input_[y].reshape(-1, *shape).float().to(device)
                label_temp = torch.argmax(label_[y])

                attribution = attribute_tec.attribute(input_temp, target=label_temp, **kwargs)
                attributions.extend(attribution)

                del attribution

        attributions = torch.stack(attributions)
        attributions_train[at_name] = attributions.detach().cpu().reshape(-1, shape[-1]).numpy()

        del attributions

        preds = torch.stack(preds)
        predictions_train[at_name] = preds.cpu().detach().numpy()

        del preds

        del attribute_tec
        torch.cuda.empty_cache()
    
    return attributions_train, predictions_train


