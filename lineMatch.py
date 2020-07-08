def FLD(image):
    # Create default Fast Line Detector class
    fld = cv2.ximgproc.createFastLineDetector()
    # Get line vectors from the image
    lines = fld.detect(image)
    # Draw lines on the image
    line_on_image = fld.drawSegments(image, lines)
    # Plot
    plt.imshow(line_on_image, interpolation='nearest', aspect='auto')
    plt.show()
    return line_on_image
