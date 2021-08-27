import paddle


def clip_pad_images(tensor, pad_shape, pad=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, paddle.Tensor):
        tensor = paddle.to_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = paddle.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def clip_pad_boxes(tensor, pad_length, pad=0):
    """
        Clip boxes of the pad area.
        :param tensor: [k, d]
        :param pad_shape: K
        :return: [K, d]
    """
    if not isinstance(tensor, paddle.Tensor):
        tensor = paddle.to_tensor(tensor)
    k = tensor.shape[0]
    d = tensor.shape[1]
    K = pad_length
    tensor_ret = paddle.zeros((K, d), dtype=tensor.dtype) + pad
    tensor_ret[:min(k, K), :] = tensor[:min(k, K), :]

    return tensor_ret


def clip_pad_1d(tensor, pad_length, pad=0):
    # dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    if not isinstance(tensor, paddle.Tensor):
        tensor = paddle.to_tensor(tensor, dtype='int64')
    tensor_ret = paddle.zeros((pad_length,), dtype=tensor.dtype) + pad
    if tensor.shape[0] > 0:
        truncate_len = min(tensor.shape[0], pad_length)
        tensor_ret[:truncate_len] = tensor[:truncate_len]
    return tensor_ret


def clip_pad_2d(tensor, pad_shape, pad=0):
    if not isinstance(tensor, paddle.Tensor):
        tensor = paddle.to_tensor(tensor)
    tensor_ret = paddle.zeros(*pad_shape, dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
        = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]

    return tensor_ret
