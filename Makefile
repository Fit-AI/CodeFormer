all: codeformer

environment:
ifndef SUPABASE_KEY
	$(error SUPABASE_KEY is undefined)
endif
ifndef SUPABASE_URL
	$(error SUPABASE_URL is undefined)
endif

codeformer: environment
	docker build -t gcr.io/savvy-webbing-347620/codeformer-api-vertex:latest \
				 --build-arg SUPABASE_URL_ARG=${SUPABASE_URL} 	\
				 --build-arg SUPABASE_KEY_ARG=${SUPABASE_KEY} 	\
				 .

install: codeformer
	docker push gcr.io/savvy-webbing-347620/codeformer-api-vertex:latest
