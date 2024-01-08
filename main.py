

from BlobAnimation import BlobAnimation
from BlobEntityTracker import BlobEntityTracker
from JPDA import JPDA


def main():
    animation = BlobAnimation()

    animation.detector.detect_blobs()
    blobs = animation.detector.get_blobs()

    if blobs:
        # Generuj kolory tylko raz
        colors = [(0,255,0)]

        tracker = BlobEntityTracker(blobs, colors)
        tracker.track_entities()

        # Przekazuj te same kolory za każdym razem, gdy tworzysz nowy obiekt BlobAnimation
        animation = BlobAnimation(tracker, color=colors)
        animation.run()

        jpda = JPDA(blobs)
        jpda.calculate_probabilities_main()
        tracks = jpda.get_tracks()
    else:
        print("No blobs detected.")

if __name__ == "__main__":
    main()