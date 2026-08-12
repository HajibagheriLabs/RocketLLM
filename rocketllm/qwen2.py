


from .base import RocketModel



class RocketQWen2(RocketModel):


    def __init__(self, *args, **kwargs):


        super(RocketQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


