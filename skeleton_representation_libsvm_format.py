import numpy
import os

bin_size = 20
def getActionNumber(file_name): #find number between a and _ in the file name
    #print(file_name)
    start = file_name.find('/a')
    end = file_name.find('_')
    action = file_name[start+2:end]
    #print(action)
    return int(action)

def getAngle(a,b,c):

    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)

    if numpy.sum(a) == 0.0:
        return 0
    if numpy.sum(b) == 0.0:
        return 0
    if numpy.sum(c) == 0.0:
        return 0

    ba = a - b
    bc = c - b

    if numpy.sum(ba) == 0.0:
        return 0
    if numpy.sum(bc) == 0.0:
        return 0

    cosine_angle = numpy.dot(ba, bc) / (numpy.linalg.norm(ba) * numpy.linalg.norm(bc))
    angle = numpy.arccos(cosine_angle)

    return numpy.degrees(angle)

def convert_to_hjdp_from_file(file_name):
    #read in all joints from file and put into a dictionary
    f = open(file_name,"r")
    relevant_joint_dictionary = {}
    for joint_information_string in f:
        #add relevant joints to a dictionary by frame and joint id
        #print(joint_information_string)
        j_info = joint_information_string.split()
        #crash prevention
        for i in range(5):
            if j_info[i] == "NaN":
                j_info[i]=1
        j_info[0] = (int)(j_info[0])
        j_info[1] = (int)(j_info[1])
        j_info[2] = (float)(j_info[2])
        j_info[3] = (float)(j_info[3])
        j_info[4] = (float)(j_info[4])

        relevant_joint_dictionary[(j_info[0],j_info[1])] = (j_info[2],j_info[3], j_info[4])
    f.close()

    #create dictionary to hold all joint values
    relative_distance = {}
    for i in range(2,21):#[2,20]
        relative_distance[i]=[] #initialize dictionary

    #fill dictionary
    for key in relevant_joint_dictionary:
        frame = key[0]
        joint_id = key[1]
        if joint_id ==1:#break if measuring centroid to centroid
            continue
        #calculate distance
        a = relevant_joint_dictionary[(frame,joint_id)]
        b = relevant_joint_dictionary[(frame,1)]
        a = numpy.array(a)
        b = numpy.array(b)
        dist = numpy.linalg.norm(a-b)
        relative_distance[joint_id].append(dist)
    return relative_distance

def convert_to_rad_from_file(file_name):
    f = open(file_name,"r")
    relevant_joint_dictionary = {} #initialize as empty dictionary will have (frame,joint) as key (x,y,z) as value
    for joint_information_string in f:
        #add relevant joints to a dictionary by frame and joint id
        #print(joint_information_string)
        j_info = joint_information_string.split()
        #crash prevention
        for i in range(5):
            if j_info[i] == "NaN":
                j_info[i]=1
        j_info[0] = (int)(j_info[0])
        j_info[1] = (int)(j_info[1])
        j_info[2] = (float)(j_info[2])
        j_info[3] = (float)(j_info[3])
        j_info[4] = (float)(j_info[4])

        #print(j_info)
        if(j_info[1] in [1,4,8,12,16,20]):#joint is relevant
            #print((j_info[0],j_info[1]))
            relevant_joint_dictionary[(j_info[0],j_info[1])] = (j_info[2],j_info[3], j_info[4])
    f.close()

    #debug test
    if(len(relevant_joint_dictionary)%6 != 0 or len(relevant_joint_dictionary)==0):
        print("Error, invalid dictionary of size "+str(len(relevant_joint_dictionary)))

    #iterate through relevant joints by frame
    distances = {
        "head":[],
        "left_arm":[],
        "right_arm":[],
        "left_leg":[],
        "right_leg":[],
    }
    angles = {
        "top_right":[],
        "bottom_right":[],
        "bottom":[],
        "bottom_left":[],
        "top_left":[],
    }

    for frame in range(1,1+(int)(len(relevant_joint_dictionary)/6)):#for each frame
        for joint_id in [4,8,12,16,20]:#for each joint in the frame
            body_part = get_body_part_rad(joint_id)
            if(body_part == "none"):
                print("Error, no joint found with the number" + str(joint_id))
            #calculate distances for each joint
            a = relevant_joint_dictionary[(frame,joint_id)]
            b = relevant_joint_dictionary[(frame,1)]
            a = numpy.array(a)
            b = numpy.array(b)
            joint_length = numpy.linalg.norm(a-b)
            distances[body_part].append(joint_length)
        #calculate angles
        center = relevant_joint_dictionary[(frame,1)]
        head = relevant_joint_dictionary[(frame,4)]
        left_arm = relevant_joint_dictionary[(frame,8)]
        right_arm = relevant_joint_dictionary[(frame,12)]
        left_leg = relevant_joint_dictionary[(frame,16)]
        right_leg = relevant_joint_dictionary[(frame,20)]

        #all 5 angles
        angles["top_right"].append(getAngle(head,center,right_arm))
        angles["bottom_right"].append(getAngle(right_arm,center,right_leg))
        angles["bottom"].append(getAngle(left_leg,center,right_leg))
        angles["bottom_left"].append(getAngle(left_arm,center,left_leg))
        angles["top_left"].append(getAngle(head,center,left_arm))
    return (distances,angles)


def get_body_part_rad(joint_id):
    part={
        4:"head",
        8:"left_arm",
        12:"right_arm",
        16:"left_leg",
        20:"right_leg",
    }
    return part.get(joint_id,"none")

def arrays_to_histograms(dist,ang,n,m,t):
    file_line = []
    for key in dist:
        #file_line.append(numpy.histogram(dist[key],n)[0])
        file_line.extend(numpy.histogram(dist[key],n)[0])
    for key in ang:
        file_line.extend(numpy.histogram(ang[key],m)[0])
    #file_line = file_line/t
    #return file_line
    return [x*(1.0/t) for x in file_line] #return concatenated histograms adjusted for number of frames

def hjdp_hist_to_libsvm(output_name,mode):
    print("Converting raw dataset'"+ mode +"' HJDP histogram in libsvm compatible file  -> "+ output_name)
    #RAD Training Set
    file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    file_name_list = file_name_list.split()
    #print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    print("Starting with "+ file_name_list[0])
    print(". . .")

    output_file = open("representations/"+output_name,"w")
    file_count = 1
    for file_name in file_name_list:
        distances = convert_to_hjdp_from_file(file_name)#convert file to two dictionaries of relevant data
        file_histogram_concatinated = arrays_to_histograms(distances,{},bin_size,1,len(distances[2])) #convert relevant data to concatinated set of histograms
        #D2 Code
        action_label = getActionNumber(file_name)
        output_file.write(str(action_label)+" ")#write lable

        # for element in file_histogram_concatinated:
        #     output_file.write(str(element)+" ")
        for i in range(len(file_histogram_concatinated)):
            output_file.write(str(i+1)+':'+str(file_histogram_concatinated[i])+' ')
        output_file.write("\n")
        if(file_count%5==0 or file_count == len(file_name_list)):
            print(str(file_count)+":"+file_name+" Completed")
        file_count = file_count + 1
    output_file.close()
    print()
    print()

def rad_hist_to_libsvm(output_name,mode):
    print("Converting raw dataset'"+ mode +"' RAD histogram in libsvm compatible file  -> "+ output_name)
    #RAD Training Set
    file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    file_name_list = file_name_list.split()
    #print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    print("Starting with "+ file_name_list[0])
    print(". . .")

    output_file = open("representations/"+output_name,"w")
    file_count = 1
    for file_name in file_name_list:
        distances, angles = convert_to_rad_from_file(file_name)#convert file to two dictionaries of relevant data
        file_histogram_concatinated = arrays_to_histograms(distances,angles,bin_size,bin_size,len(distances["head"])) #convert relevant data to concatinated set of histograms
        #D2 Code
        action_label = getActionNumber(file_name)
        output_file.write(str(action_label)+" ")#write lable

        # for element in file_histogram_concatinated:
        #     output_file.write(str(element)+" ")
        for i in range(len(file_histogram_concatinated)):
            output_file.write(str(i+1)+':'+str(file_histogram_concatinated[i])+' ')
        output_file.write("\n")
        if(file_count%5==0 or file_count == len(file_name_list)):
            print(str(file_count)+":"+file_name+" Completed")
        file_count = file_count + 1
    output_file.close()
    print()
    print()

def main():
    print("Extracting Files")
    #RAD Training Set
    rad_hist_to_libsvm("rad_d2","train")
    rad_hist_to_libsvm("rad_d2.t","test")

    #HJDP Training set
    hjdp_hist_to_libsvm("hjdp_d2","train")
    hjdp_hist_to_libsvm("hjdp_d2.t","test")

    #Training and evaluation


    # mode="train"
    # file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    # file_name_list = file_name_list.split()
    # print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    # print("Starting with "+ file_name_list[0])
    # print(". . .")
    #
    # output_file = open("representations/rad_d2","w")
    # file_count = 1
    # for file_name in file_name_list:
    #     distances, angles = convert_to_rad_from_file(file_name)#convert file to two dictionaries of relevant data
    #     file_histogram_concatinated = arrays_to_histograms(distances,angles,bin_size,bin_size,len(distances["head"])) #convert relevant data to concatinated set of histograms
    #     #D2 Code
    #     action_label = getActionNumber(file_name)
    #     output_file.write(str(action_label)+" ")#write lable
    #
    #     # for element in file_histogram_concatinated:
    #     #     output_file.write(str(element)+" ")
    #     for i in range(len(file_histogram_concatinated)):
    #         output_file.write(str(i+1)+':'+str(file_histogram_concatinated[i])+' ')
    #     output_file.write("\n")
    #     if(file_count%5==0 or file_count == len(file_name_list)):
    #         print(str(file_count)+":"+file_name+" Completed")
    #     file_count = file_count + 1
    # output_file.close()
    # print()
    # print()

    # #Rad Testing Set
    # mode="test"
    # file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    # file_name_list = file_name_list.split()
    # print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    # print("Starting with "+ file_name_list[0])
    # print(". . .")
    #
    # output_file = open("representations/rad_d2.t","w")
    # file_count = 1
    # for file_name in file_name_list:
    #     distances, angles = convert_to_rad_from_file(file_name)#convert file to two dictionaries of relevant data
    #     file_histogram_concatinated = arrays_to_histograms(distances,angles,bin_size,bin_size,len(distances["head"])) #convert relevant data to concatinated set of histograms
    #     for element in file_histogram_concatinated:
    #         output_file.write(str(element)+" ")
    #     output_file.write("\n")
    #     if(file_count%5==0 or file_count == len(file_name_list)):
    #         print(str(file_count)+":"+file_name+" Completed")
    #     file_count = file_count + 1
    # output_file.close()
    # print()
    # print()
    #
    #
    #
    # print("Extracting files for HJDP Set")
    # #HJDP Training Set
    # mode="train"
    # file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    # file_name_list = file_name_list.split()
    # print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    # print("Starting with "+ file_name_list[0])
    # print(". . .")
    #
    # output_file = open("representations/hjdp_d2","w")
    # file_count = 1
    # for file_name in file_name_list:
    #     distances = convert_to_hjdp_from_file(file_name)#convert file to two dictionaries of relevant data
    #     file_histogram_concatinated = arrays_to_histograms(distances,{},bin_size,1,len(distances[2])) #convert relevant data to concatinated set of histograms
    #     for element in file_histogram_concatinated:
    #         output_file.write(str(element)+" ")
    #     output_file.write("\n")
    #     if(file_count%5==0 or file_count == len(file_name_list)):
    #         print(str(file_count)+":"+file_name+" Completed")
    #     file_count = file_count + 1
    # output_file.close()
    # print()
    #
    # #HJDP Testing Set
    # mode="test"
    # file_name_list = os.popen("ls dataset/"+mode+"/*").read()#os.system("ls dataset/"+mode+"/*")
    # file_name_list = file_name_list.split()
    # print("There are "+str(len(file_name_list))+" files to be converted from the " + mode + " set of data")
    # print("Starting with "+ file_name_list[0])
    # print(". . .")
    #
    # output_file = open("representations/hjdp_d2.t","w")
    # file_count = 1
    # for file_name in file_name_list:
    #     distances = convert_to_hjdp_from_file(file_name)#convert file to two dictionaries of relevant data
    #     file_histogram_concatinated = arrays_to_histograms(distances,{},bin_size,1,len(distances[2])) #convert relevant data to concatinated set of histograms
    #     for element in file_histogram_concatinated:
    #         output_file.write(str(element)+" ")
    #     output_file.write("\n")
    #     if(file_count%5==0 or file_count == len(file_name_list)):
    #         print(str(file_count)+":"+file_name+" Completed")
    #     file_count = file_count + 1
    # output_file.close()
    print("DONE")
    print()
    #
    #
    #
    #
    #
    # #distances, angles = convert_to_rad_from_file("dataset/train/a08_s01_e01_skeleton_proj.txt")
    #
    # #for key in distances:
    #     #print(len(distances[key]))
    #     #print(numpy.histogram(distances[key],5)[0])
    #     #print(numpy.histogram(distances[key],5)[1])
    #     #print()
    # #print()
    # #for key in angles:
    #     #print(numpy.histogram(angles[key],5)[0])
    #     #print(numpy.histogram(angles[key],5)[1])
    #     #print()
    #     #print(len(angles[key]))
    #     #for dist in distances[key]:
    #     #    print(dist)
    # #print(len(distances["head"]))
    # #print(len(angles["top_right"]))
    # print()
    # #histogram_1 = arrays_to_histograms(distances,angles,5,5,len(distances["head"]))
    # #print(histogram_1)

if __name__ == "__main__":
        main()
