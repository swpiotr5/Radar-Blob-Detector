class AssociationManager:
    @staticmethod
    def associate_blobs(previous_probabilities, current_probabilities):
        associations = []

        for j in range(len(current_probabilities)):
            max_probability = -1
            best_match = -1

            for k in range(len(previous_probabilities)):
                probability_ij = current_probabilities[j]
                prev_prob = previous_probabilities[k]

                prob = prev_prob * probability_ij
                if prob > max_probability:
                    max_probability = prob
                    best_match = k

            associations.append((j, best_match))

        return associations