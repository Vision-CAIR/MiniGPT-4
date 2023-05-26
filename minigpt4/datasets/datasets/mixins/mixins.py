from collections import OrderedDict


class __ImageDisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "vision": sample["vision"],
            }
        )


class __AudioDisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        # TODO: Finish the Audio Display Mixin
        '''
        return OrderedDict(
            {
            }
        )
        '''

        raise NotImplementedError

