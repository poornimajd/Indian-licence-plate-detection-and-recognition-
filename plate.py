def get_chars(x11, x1, y1, x2, y2, scores, labels, classes):
    chars = []
    aa = []
    xx1 = list(x1)
    while(len(x1) != 0):
        try:
            boxA = [min(x1), y1[x1.index(min(x1))],
                    x2[x1.index(min(x1))], y2[x1.index(min(x1))]]
            boxB = [x1_char, y1[inde], x2[inde], y2[inde]]
            iou = calculate_iou(boxA, boxB)
            if iou < 0.8:
                x1_char = min(x1)

                name = labels[x11.index(x1_char)]
                chars.append(name)
                inde = x11.index(x1_char)
                # print("c", inde)
                x1.remove(x1_char)
            elif scores[inde] > scores[x11.index(min(x1))]:
                x1.remove(min(x1))
                continue
            else:
                chars.remove(chars[-1])
                x1_char = min(x1)

                name = labels[x11.index(x1_char)]
                chars.append(name)
                inde = x11.index(x1_char)
                # print("c", inde)
                x1.remove(x1_char)

        except Exception as e:
            print(e)

            x1_char = min(x1)

            # print(labels[x11.index(x1_char)])
            name = labels[x11.index(x1_char)]
            chars.append(name)
            aa.append(x1_char)
            inde = x11.index(x1_char)
            # print("c", inde)
            x1.remove(x1_char)

    for jj in range(len(aa)-1):
        print(abs(aa[jj]-aa[jj+1]))
        if abs(aa[jj]-aa[jj+1]) < 0.01:

            if scores[x11.index(aa[jj])] > scores[x11.index(aa[jj+1])]:

                ss = (aa.index(aa[jj+1]))
            else:
                ss = (aa.index(aa[jj]))
    try:
        chars[ss] = '#'
        chars.remove('#')
    except:
        print("NO repetation")
    return chars


def coordinates(x1, y1, x2, y2, height, scores, labels, classes):
    # print("coordinates", x1, y1)
    if len(x1) != 0 or len(y1) != 0:
        for i in y1:
            if y1.count(i) > 1:
                indices = [p for p, v in enumerate(y1) if v == i]
                y1[indices[0]] += 0.001*(y1.count(i))
        for j in x1:
            if x1.count(j) > 1:
                indices = [p for p, v in enumerate(x1) if v == j]
                x1[indices[0]] += 0.001*(x1.count(j))
        yy1 = list(y1)
        xx1 = list(x1)
        x11 = list(x1)
        # print("coordinated", x1, y1)
        y1_max = max(y1)
        y1_min = min(y1)
        top_chars = []
        chars = []
        threshold = height
        if abs(y1_max - y1_min) < threshold:
            chars = get_chars(x11, x1, y1, x2, y2, scores, labels, classes)

        else:
            top_chars.append(xx1[yy1.index(y1_min)])
            print("x", (y1_min), " ", yy1.index(
                y1_min), " ", xx1[yy1.index(y1_min)])
            y1.remove(y1_min)
            x1.remove(xx1[yy1.index(y1_min)])
            while(len(y1) != 0):
                y1_min = min(y1)
                if abs(y1_max - y1_min) < threshold:
                    break
                else:
                    top_chars.append(xx1[yy1.index(y1_min)])

                    # print("x", (y1_min), " ", yy1.index(
                    #     y1_min), " ", xx1[yy1.index(y1_min)])
                    y1.remove(y1_min)
                    # print(xx1[yy1.index(y1_min)])
                    x1.remove(xx1[yy1.index(y1_min)])

            # print("top", top_chars, x1)
            chars.extend(get_chars(x11, top_chars, y1,
                                   x2, y2, scores, labels, classes))
            chars.extend(get_chars(x11, x1, y1, x2,
                                   y2, scores, labels, classes))
        o = [2, 3]
        o.extend([i for i in range(len(chars)-4, len(chars))])
        try:
            for i in o:
                if chars[i] == 'O':
                    chars[i] = '0'
                elif chars[i] == 'Z':
                    chars[i] = "2"
                elif chars[i] == "I":
                    chars[i] = "1"
        except:
            print("Length of the string is less than 4")
        return chars


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
