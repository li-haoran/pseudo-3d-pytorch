import os

def partial(data=''):
    file=os.listdir(data)
    for f in file:
        if f.startswith('hmdb'):
            print f
            a=[]
            fname=os.path.join(data,f)
            with open(fname,'r') as fd:
                for x in fd:
                    xi=x.strip().split(' ')
                    name=xi[0].split('/')[-1]
                    n=xi[1]
                    c=xi[2]
                    s='%s %s %s\n'%(name,n,c)
                    a.append(s)

            this_f='refine'+f
            fname=os.path.join(data,this_f)
            with open(fname,'w') as pd:
                pd.writelines(a)


if __name__=='__main__':
    partial('dataset')