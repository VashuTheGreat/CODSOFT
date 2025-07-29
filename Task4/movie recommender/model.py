class MoviePredictorModel:
    def compile(self, new_movie):
        self.new_movie = new_movie

    def fit(self, similarities):
        self.similarities = similarities

    def predict(self, name, num):
        mov_rec = []
        index = self.new_movie[self.new_movie['title'].str.lower() == name.lower()].index[0]
        rec = sorted(enumerate(self.similarities[index]), key=lambda x: x[1], reverse=True)[1:num + 1]
        for i in rec:
            mov_rec.append((self.new_movie.iloc[i[0]].movie_id, self.new_movie.iloc[i[0]].title))
        return mov_rec


