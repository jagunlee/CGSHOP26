parallel_flip_path(), parallel_flip_rev():
    while True:
        ...
        _, score1 = cand[0]
        _, score2 = cand[-1]

        highest_score = max(score1[0], highest_score)
        lowest_score = min(score2[0], lowest_score)

        cur_score = [s[0] for _, s in cand]
        cur_mean = sum(cur_score)/len(cand)
        cur_std = math.sqrt(sum([(cs-cur_mean)**2 for cs in cur_score])/len(cand))
        mean_score.append(cur_mean)
        std_score.append(cur_std)
        ...
