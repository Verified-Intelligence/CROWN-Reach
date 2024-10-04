From yuhaochrisj/crown-reach-env:1.2


COPY CROWN-Reach /home/CROWN-Reach
RUN chmod -R 777 /home/CROWN-Reach

RUN git clone https://github.com/chenxin415/flowstar.git /home/CROWN-Reach/flowstar && \
    cd /home/CROWN-Reach/flowstar/flowstar-toolbox && \
    make

RUN find /home/CROWN-Reach/archcomp -type f -name 'Makefile' -execdir make \;

COPY run.py /home/run.py
RUN chmod 777 /home/run.py