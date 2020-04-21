import numpy

def getAngle(a,b,c):
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)

    ba = a - b
    bc = c - b
    cosine_angle = numpy.dot(ba, bc) / (numpy.linalg.norm(ba) * numpy.linalg.norm(bc))
    angle = numpy.arccos(cosine_angle)

    return numpy.degrees(angle)

def convert_to_rad_from_file(file_name):
    f = open(file_name,"r")
    relevant_joint_dictionary = {} #initialize as empty dictionary will have (frame,joint) as key (x,y,z) as value
    for joint_information_string in f:
        #add relevant joints to a dictionary by frame and joint id
        #print(joint_information_string)
        j_info = joint_information_string.split()
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

def main():
    print("extracting files")
    distances, angles = convert_to_rad_from_file("dataset/train/a08_s01_e01_skeleton_proj.txt")
    for key in distances:
        #print(len(distances[key]))
        numpy.histogram(distances[key],5)
    print()
    for key in angles:
        print(len(angles[key]))
        #for dist in distances[key]:
        #    print(dist)
    #print(len(distances["head"]))
    #print(len(angles["top_right"]))

if __name__ == "__main__":
        main()
