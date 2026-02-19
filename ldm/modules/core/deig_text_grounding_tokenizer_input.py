import torch as th


class DEIGGroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the
        input only for the ground tokenizer.
        batch = {
            'subject_boxes': [[..],[..],..]],
            'object_boxes': [[..],[..],..]],
            'masks': ..,
            'subject_text_embeddings': [..],
            'object_text_embeddings': [..],
            'action_text_embeddings': [..]
        }
        """

        self.set = True

        object_boxes = batch['object_boxes']
        masks = batch['masks']
        object_positive_embeddings = batch["object_text_embeddings"]
        inst_masks = batch["inst_masks"]

        self.batch, self.max_box, self.num_latents, self.in_dim = object_positive_embeddings.shape
        self.device = object_positive_embeddings.device
        self.dtype = object_positive_embeddings.dtype

        return {"object_boxes": object_boxes,
                "masks": masks,
                "object_positive_embeddings": object_positive_embeddings,
                "inst_masks": inst_masks
        }

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        object_boxes = th.zeros(batch, self.max_box, 4, ).type(dtype).to(device)
        masks = th.zeros(batch, self.max_box, 1).type(dtype).to(device)
        object_positive_embeddings = th.zeros(batch, self.max_box, self.num_latents, self.in_dim).type(dtype).to(device)
        object_attn_masks = object_attn_masks = th.zeros(batch, self.max_box, 512, 512).type(dtype).to(device)

        return {"object_boxes": object_boxes,
                "masks": masks,
                "object_positive_embeddings": object_positive_embeddings,
                "inst_masks": object_attn_masks}
