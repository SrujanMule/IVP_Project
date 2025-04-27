# Predict

def classify_galaxy(list_of_galaxy_img_paths):
    """
    Loads a list of image paths (from the data/ directory), preprocesses them
    into a batch tensor suitable for the model, and returns predictions.

    Args:
        list_of_galaxy_img_paths (List[str]): Paths to JPEG image files.

    Returns:
        List[Dict]: List of dictionaries containing:
            - "path": str, original image path
            - "probabilities": List[float], 37-class normalized probability vector
            - "predicted_class": str, predicted class label ("1.1", "1.2", or "1.3")
    """
    import torch
    from PIL import Image
    from interface import TheModel, the_dataloader, TheDataset, the_batch_size
    from dataset import get_transform
    from config import weights_path, device

    # Mapping from index to class label
    class_labels = ["1.1", "1.2", "1.3"]

    # Convert to input suitable for the model
    transform = get_transform()
    galaxy_batch = []
    for path in list_of_galaxy_img_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        galaxy_batch.append(tensor)

    galaxy_batch = torch.stack(galaxy_batch).to(device)

    # Load the model
    model = TheModel(num_classes=37).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Predict the outcome
    with torch.no_grad():
        logits = model(galaxy_batch)

    logits_cpu = logits.cpu()
    probabilities = logits_cpu.numpy().tolist()

    # Extract first three logits and compute argmax for each galaxy
    first_three_logits = logits_cpu[:, :3]
    predicted_indices = torch.argmax(first_three_logits, dim=1).tolist()
    predicted_classes = [class_labels[idx] for idx in predicted_indices]

    # Package everything into a list of dictionaries
    results = []
    for path, prob, pred_class in zip(list_of_galaxy_img_paths, probabilities, predicted_classes):
        results.append({
            "path": path,
            "probabilities": prob,
            "predicted_class": pred_class
        })

    return results
