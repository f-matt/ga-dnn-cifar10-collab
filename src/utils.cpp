#include "utils.h"

/*
 * Memory usage (current process)
 */
int get_memory_usage() {
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL){
		if (strncmp(line, "VmRSS:", 6) == 0){
			// This assumes that a digit will be found and the line ends in " Kb".
			int i = strlen(line);
			const char* p = line;
			while (*p <'0' || *p > '9') p++;
			line[i-3] = '\0';
			result = atoi(p);

			break;
		}
	}

	fclose(file);
	return result;

}
