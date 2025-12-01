import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정 (.env 파일의 1-3줄 형식 사용)
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o")
azure_oai_api_version = os.getenv("AZURE_OAI_API_VERSION", "2024-02-01")

# 11개 국가별 넷플릭스 인기 작품 순위 및 캐릭터 데이터베이스
NETFLIX_DATA = {
    "한국": {
        "작품": {
            "오징어 게임": {
                "순위": 1,
                "카테고리": "스릴러/서바이벌",
                "별점": 4.8,
                "유명_명대사": "우리는 모두 같은 배를 탔어",
                "캐릭터": {
                    "성기훈": {
                        "성격": "도박 중독에 빠진 전직 자동차 공장 직공. 실패한 사업가이지만 가족에 대한 사랑이 깊고, 생존을 위해 끝까지 싸우는 강인한 의지를 가진 인물이에요!",
                        "특징": "인간적이고 현실적인 모습을 보여주며, 극한 상황에서도 동료를 배려하는 따뜻한 마음을 가지고 있어요.",
                        "명대사": "우리는 모두 같은 배를 탔어",
                        "능력치": {"지능": 7, "체력": 6, "정신력": 9, "리더십": 8, "전략": 7}
                    },
                    "조상우": {
                        "성격": "서울대 출신의 똑똑한 투자 전문가. 냉정하고 계산적이지만, 마지막 순간에는 인간미를 보여주는 복합적인 캐릭터예요!",
                        "특징": "높은 지능과 전략적 사고를 가지고 있지만, 인간관계에서는 냉담한 편이에요.",
                        "명대사": "이 게임은 공정하지 않아",
                        "능력치": {"지능": 10, "체력": 5, "정신력": 8, "리더십": 6, "전략": 10}
                    }
                }
            },
            "더 글로리": {
                "순위": 2,
                "카테고리": "복수/드라마",
                "별점": 4.7,
                "유명_명대사": "나는 복수를 할 거예요",
                "캐릭터": {
                    "문동은": {
                        "성격": "학교 폭력의 피해자였지만, 18년간 치밀하게 복수 계획을 세워 실행하는 냉정하고 강인한 여성예요!",
                        "특징": "매우 똑똑하고 인내심이 강하며, 목표를 향해 한 치의 흔들림도 없는 집중력을 가지고 있어요.",
                        "명대사": "나는 복수를 할 거예요",
                        "능력치": {"지능": 10, "체력": 4, "정신력": 10, "리더십": 7, "전략": 10}
                    },
                    "주여정": {
                        "성격": "문동은의 복수 계획을 돕는 의사. 정의감이 강하고, 동은을 진심으로 이해하고 지지하는 따뜻한 인물이에요!",
                        "특징": "의료진으로서의 전문성과 인간에 대한 깊은 공감 능력을 동시에 가지고 있어요.",
                        "명대사": "당신의 편이 될게요",
                        "능력치": {"지능": 9, "체력": 6, "정신력": 9, "리더십": 8, "전략": 8}
                    }
                }
            },
            "스위트홈": {
                "순위": 3,
                "카테고리": "호러/액션",
                "별점": 4.5,
                "유명_명대사": "우리는 괴물이 아니야",
                "캐릭터": {
                    "현수진": {
                        "성격": "고등학생이지만 괴물화된 세상에서 살아남기 위해 강인하게 싸우는 용감한 청년이에요!",
                        "특징": "처음에는 소심했지만, 위기를 겪으며 점점 성장하고 리더십을 발휘하는 인물이에요.",
                        "명대사": "우리는 괴물이 아니야",
                        "능력치": {"지능": 7, "체력": 9, "정신력": 8, "리더십": 9, "전략": 7}
                    }
                }
            },
            "킹덤": {
                "순위": 4,
                "카테고리": "사극/좀비",
                "별점": 4.6,
                "유명_명대사": "백성들이 굶주리고 있는데, 나는 무엇을 하고 있었나",
                "캐릭터": {
                    "이창": {
                        "성격": "조선의 왕세자로, 백성들을 위해 목숨을 걸고 싸우는 정의로운 인물이에요!",
                        "특징": "무예에 뛰어나고, 백성들의 고통을 이해하며 함께 고민하는 진정한 리더예요.",
                        "명대사": "백성들이 굶주리고 있는데, 나는 무엇을 하고 있었나",
                        "능력치": {"지능": 8, "체력": 9, "정신력": 9, "리더십": 10, "전략": 8}
                    }
                }
            }
        }
    },
    "미국": {
        "작품": {
            "기묘한 이야기": {
                "순위": 1,
                "카테고리": "SF/호러",
                "별점": 4.9,
                "유명_명대사": "Friends don't lie",
                "캐릭터": {
                    "일레븐": {
                        "성격": "초능력을 가진 소녀로, 처음에는 말이 없고 조용하지만 친구들과의 우정을 통해 점점 밝아지는 캐릭터예요!",
                        "특징": "강력한 초능력을 가지고 있지만, 평범한 삶을 꿈꾸는 순수한 소녀예요.",
                        "명대사": "Friends don't lie",
                        "능력치": {"지능": 7, "체력": 6, "정신력": 10, "리더십": 7, "전략": 8}
                    },
                    "스티브 해링턴": {
                        "성격": "처음에는 인기 있는 왕따였지만, 점점 성장하며 진정한 친구가 되는 캐릭터예요!",
                        "특징": "외모는 멋있지만 처음에는 이기적이었지만, 시간이 지나며 따뜻하고 책임감 있는 인물로 변해요.",
                        "명대사": "I may be a pretty shitty boyfriend, but turns out I'm actually a pretty damn good babysitter",
                        "능력치": {"지능": 6, "체력": 8, "정신력": 8, "리더십": 9, "전략": 7}
                    }
                }
            },
            "브리저튼": {
                "순위": 2,
                "카테고리": "로맨스/사극",
                "별점": 4.6,
                "유명_명대사": "I will not be defined by my marriage",
                "캐릭터": {
                    "다프네 브리저튼": {
                        "성격": "독립적이고 똑똑한 여성으로, 사회의 관습에 맞서 자신의 행복을 찾는 강인한 인물이에요!",
                        "특징": "독서를 좋아하고 지적 호기심이 많으며, 진정한 사랑을 믿는 낭만적인 면도 있어요.",
                        "명대사": "I will not be defined by my marriage",
                        "능력치": {"지능": 9, "체력": 5, "정신력": 9, "리더십": 8, "전략": 8}
                    }
                }
            },
            "위쳐": {
                "순위": 3,
                "카테고리": "판타지/액션",
                "별점": 4.5,
                "유명_명대사": "Hmm",
                "캐릭터": {
                    "게롤트": {
                        "성격": "냉정하고 무뚝뚝해 보이지만, 사실은 따뜻한 마음을 가진 마법사 사냥꾼이에요!",
                        "특징": "강력한 전투력을 가지고 있으며, 정의를 위해 싸우지만 감정 표현은 서툴러요.",
                        "명대사": "Hmm",
                        "능력치": {"지능": 8, "체력": 10, "정신력": 9, "리더십": 7, "전략": 9}
                    }
                }
            }
        }
    },
    "영국": {
        "작품": {
            "더 크라운": {
                "순위": 1,
                "카테고리": "드라마/역사",
                "별점": 4.7,
                "유명_명대사": "Duty first, self second",
                "캐릭터": {
                    "엘리자베스 2세": {
                        "성격": "영국 여왕으로, 국가와 국민을 위해 자신의 개인적 욕구를 억제하는 강인하고 책임감 있는 인물이에요!",
                        "특징": "매우 냉정하고 이성적이며, 전통과 의무를 중시하는 리더예요.",
                        "명대사": "Duty first, self second",
                        "능력치": {"지능": 9, "체력": 7, "정신력": 10, "리더십": 10, "전략": 9}
                    }
                }
            }
        }
    },
    "독일": {
        "작품": {
            "다크": {
                "순위": 1,
                "카테고리": "SF/스릴러",
                "별점": 4.8,
                "유명_명대사": "The question is not where, but when",
                "캐릭터": {
                    "요나스 칸발트": {
                        "성격": "시간 여행의 중심에 있는 소년으로, 진실을 찾기 위해 고군분투하는 인물이에요!",
                        "특징": "논리적이고 분석적이며, 복잡한 시간 여행의 수수께끼를 풀어가는 지능적인 캐릭터예요.",
                        "명대사": "The question is not where, but when",
                        "능력치": {"지능": 10, "체력": 6, "정신력": 9, "리더십": 8, "전략": 10}
                    }
                }
            }
        }
    },
    "스페인": {
        "작품": {
            "라 카사 데 파펠": {
                "순위": 1,
                "카테고리": "범죄/스릴러",
                "별점": 4.9,
                "유명_명대사": "Bella ciao",
                "캐릭터": {
                    "교수": {
                        "성격": "은행 강도 작전을 계획하는 천재적인 두뇌. 냉정하고 계산적이지만 동료들을 아끼는 인물이에요!",
                        "특징": "매우 똑똑하고 전략적이며, 완벽한 계획을 세우는 능력이 뛰어나요.",
                        "명대사": "Bella ciao",
                        "능력치": {"지능": 10, "체력": 5, "정신력": 9, "리더십": 10, "전략": 10}
                    }
                }
            }
        }
    },
    "프랑스": {
        "작품": {
            "루핑": {
                "순위": 1,
                "카테고리": "SF/스릴러",
                "별점": 4.6,
                "유명_명대사": "Time is a loop",
                "캐릭터": {
                    "로만": {
                        "성격": "시간 루프에 갇힌 남자. 반복되는 하루를 통해 진실을 찾아가는 인물이에요!",
                        "특징": "인내심이 강하고, 시간이 지나며 점점 성장하는 캐릭터예요.",
                        "명대사": "Time is a loop",
                        "능력치": {"지능": 8, "체력": 7, "정신력": 10, "리더십": 7, "전략": 9}
                    }
                }
            }
        }
    },
    "일본": {
        "작품": {
            "앨리스 인 보더랜드": {
                "순위": 1,
                "카테고리": "서바이벌/스릴러",
                "별점": 4.7,
                "유명_명대사": "Survive or die",
                "캐릭터": {
                    "아리사": {
                        "성격": "게임 세계에 갇힌 청년. 생존을 위해 냉정하게 판단하는 인물이에요!",
                        "특징": "논리적이고 냉정하며, 위기 상황에서도 침착하게 대처해요.",
                        "명대사": "Survive or die",
                        "능력치": {"지능": 9, "체력": 8, "정신력": 9, "리더십": 8, "전략": 9}
                    }
                }
            }
        }
    },
    "인도": {
        "작품": {
            "사쿠나마": {
                "순위": 1,
                "카테고리": "범죄/드라마",
                "별점": 4.5,
                "유명_명대사": "Justice will prevail",
                "캐릭터": {
                    "비크람": {
                        "성격": "경찰로, 정의를 위해 싸우는 강인한 인물이에요!",
                        "특징": "강한 정의감과 추리 능력을 가지고 있어요.",
                        "명대사": "Justice will prevail",
                        "능력치": {"지능": 8, "체력": 9, "정신력": 9, "리더십": 9, "전략": 8}
                    }
                }
            }
        }
    },
    "브라질": {
        "작품": {
            "3%": {
                "순위": 1,
                "카테고리": "SF/서바이벌",
                "별점": 4.4,
                "유명_명대사": "Only 3% will survive",
                "캐릭터": {
                    "미셸": {
                        "성격": "선발 과정을 통과하려는 강인한 여성. 정의감이 강하고 똑똑해요!",
                        "특징": "강한 의지와 지능을 가지고 있으며, 동료들을 보호하려는 마음이 있어요.",
                        "명대사": "Only 3% will survive",
                        "능력치": {"지능": 9, "체력": 8, "정신력": 9, "리더십": 9, "전략": 9}
                    }
                }
            }
        }
    },
    "멕시코": {
        "작품": {
            "나르코스": {
                "순위": 1,
                "카테고리": "범죄/드라마",
                "별점": 4.8,
                "유명_명대사": "Plata o plomo",
                "캐릭터": {
                    "파블로 에스코바르": {
                        "성격": "마약 카르텔의 보스. 카리스마가 있고 냉정하지만 가족에 대한 사랑이 깊어요!",
                        "특징": "강력한 리더십과 전략적 사고를 가지고 있으며, 매우 위험한 인물이에요.",
                        "명대사": "Plata o plomo",
                        "능력치": {"지능": 9, "체력": 7, "정신력": 8, "리더십": 10, "전략": 10}
                    }
                }
            }
        }
    },
    "이탈리아": {
        "작품": {
            "베이비": {
                "순위": 1,
                "카테고리": "범죄/드라마",
                "별점": 4.5,
                "유명_명대사": "We are the future",
                "캐릭터": {
                    "루도": {
                        "성격": "부유한 청년이지만 마약에 빠진 인물. 복잡한 내면을 가진 캐릭터예요!",
                        "특징": "매력적이지만 위험한 인물로, 자신의 선택에 고민이 많아요.",
                        "명대사": "We are the future",
                        "능력치": {"지능": 7, "체력": 6, "정신력": 6, "리더십": 7, "전략": 7}
                    }
                }
            }
        }
    }
}

def get_azure_openai_client():
    """Azure OpenAI 클라이언트 생성"""
    if not azure_oai_endpoint or not azure_oai_key:
        return None
    
    try:
        return AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version=azure_oai_api_version
        )
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 생성 오류: {str(e)}")
        return None

def get_country_rankings(category=None):
    """국가별 작품 순위 가져오기"""
    rankings = {}
    for country, data in NETFLIX_DATA.items():
        rankings[country] = []
        for series_name, series_data in data["작품"].items():
            if category is None or series_data["카테고리"] == category:
                rankings[country].append({
                    "작품명": series_name,
                    "순위": series_data["순위"],
                    "카테고리": series_data["카테고리"],
                    "별점": series_data["별점"],
                    "유명_명대사": series_data["유명_명대사"]
                })
        rankings[country].sort(key=lambda x: x["순위"])
    return rankings

def get_character_info(country, series_name, character_name=None):
    """캐릭터 정보 가져오기"""
    if country not in NETFLIX_DATA:
        return None
    
    if series_name not in NETFLIX_DATA[country]["작품"]:
        return None
    
    characters = NETFLIX_DATA[country]["작품"][series_name]["캐릭터"]
    
    if character_name:
        return characters.get(character_name)
    else:
        return characters

def format_star_rating(value, max_value=10):
    """별점 형식으로 표시"""
    filled = "⭐" * (value // 2)
    half = "✨" if value % 2 == 1 else ""
    empty = "☆" * ((max_value - value) // 2)
    return f"{filled}{half}{empty} ({value}/10)"

def format_character_response(country, series_name, character_name=None):
    """캐릭터 정보를 포맷팅"""
    series_data = NETFLIX_DATA[country]["작품"][series_name]
    characters = get_character_info(country, series_name, character_name)
    
    if not characters:
        return f"어라라~ 😅 {series_name} 시리즈의 정보를 찾을 수 없어요!"
    
    response = f"🎬 **{series_name}** ({country}) - {series_data['카테고리']}\n"
    response += f"⭐ 작품 별점: {series_data['별점']}/5.0\n"
    response += f"💬 유명 명대사: \"{series_data['유명_명대사']}\"\n\n"
    
    if character_name and character_name in characters:
        char = characters[character_name]
        response += f"## ✨ {character_name} ✨\n\n"
        response += f"### 🎭 성격\n{char['성격']}\n\n"
        response += f"### 🌟 특징\n{char['특징']}\n\n"
        response += f"### 💬 명대사\n\"{char['명대사']}\"\n\n"
        response += "### ⭐ 핵심 능력치\n"
        for ability, value in char['능력치'].items():
            response += f"- **{ability}**: {format_star_rating(value)}\n"
        response += "\n---\n\n"
    else:
        response += "### 주요 캐릭터들\n\n"
        for name, info in characters.items():
            response += f"#### 🎪 {name}\n"
            response += f"**성격**: {info['성격']}\n\n"
            response += f"**명대사**: \"{info['명대사']}\"\n\n"
            response += "**능력치**: "
            ability_str = ", ".join([f"{k}: {v}/10" for k, v in info['능력치'].items()])
            response += ability_str + "\n\n"
            response += "---\n\n"
    
    return response

def get_chat_response(user_message, conversation_history, temperature=0.8, max_tokens=1000, top_p=0.9):
    """Azure OpenAI를 사용하여 챗봇 응답 생성"""
    client = get_azure_openai_client()
    
    if not client:
        return "⚠️ 어라라~ Azure OpenAI 설정이 완료되지 않았어요! .env 파일을 확인해주세요~ 😅"
    
    system_prompt = """당신은 넷플릭스 드라마와 영화의 캐릭터 성격을 알려주는 귀엽고 깜찍한 챗봇이에요!

말투 규칙:
- 항상 "~해요", "~예요", "~어요" 같은 존댓말을 사용해요
- 이모지를 적절히 사용해서 친근하게 대답해요
- "어라라~", "와아~", "헤헤~" 같은 귀여운 감탄사를 사용해요
- 매우 친근하고 따뜻한 톤으로 대답해요

사용자가 넷플릭스 작품의 캐릭터에 대해 물어보면:
- 해당 캐릭터의 성격, 특징, 명대사, 능력치 등을 귀엽고 친근하게 설명해주세요
- 작품의 배경과 스토리도 간단히 언급해주세요
- 캐릭터의 매력 포인트를 강조해서 설명해주세요

한국어로 자연스럽고 귀엽게 대답해주세요!"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in conversation_history[-10:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"어라라~ 오류가 발생했어요! 😅 {str(e)}\n\n다시 시도해주세요~"

def extract_series_and_character(user_message):
    """사용자 메시지에서 국가, 시리즈명, 캐릭터명 추출"""
    countries = list(NETFLIX_DATA.keys())
    
    found_country = None
    found_series = None
    found_character = None
    
    for country in countries:
        if country in user_message:
            found_country = country
            for series_name in NETFLIX_DATA[country]["작품"].keys():
                if series_name in user_message:
                    found_series = series_name
                    characters = NETFLIX_DATA[country]["작품"][series_name]["캐릭터"]
                    for char_name in characters.keys():
                        if char_name in user_message:
                            found_character = char_name
                            break
                    break
            break
    
    return found_country, found_series, found_character

def main():
    st.set_page_config(
        page_title="넷플릭스 캐릭터 성격 챗봇",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎬 넷플릭스 전세계 작품 순위 & 캐릭터 챗봇 🎭</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">11개 국가별 인기 작품 순위와 캐릭터 정보를 별점으로 확인하세요! 💕</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = """안녕하세요! 🎬✨ 

저는 넷플릭스 전세계 작품 순위와 캐릭터들의 성격, 능력치를 알려주는 전문 챗봇이에요! 🕵️‍♀️💕

🌍 **11개 국가별 인기 작품 순위**
- 한국, 미국, 영국, 독일, 스페인, 프랑스, 일본, 인도, 브라질, 멕시코, 이탈리아

🎭 **캐릭터 정보**
- 성격, 특징, 명대사
- 핵심 능력치 별점 (지능, 체력, 정신력, 리더십, 전략)

💬 **작품 검색 시**
- 가장 유명한 명대사도 함께 알려드려요!

카테고리별로 전세계 작품 순위도 확인할 수 있어요! 🎪

지금 바로 물어봐주세요! 🚀"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 답변 창의성 조절")
        st.markdown("---")
        
        temperature = st.slider("Temperature (창의성)", 0.0, 1.0, 0.8, 0.1,
                               help="높을수록 더 창의적이고 다양한 답변을 생성해요")
        max_tokens = st.slider("Max Tokens (답변 길이)", 500, 2000, 1000, 100,
                              help="답변의 최대 길이를 조절해요")
        top_p = st.slider("Top P (다양성)", 0.0, 1.0, 0.9, 0.1,
                         help="높을수록 더 다양한 단어를 선택해요")
        
        st.markdown("---")
        st.header("🌍 국가별 작품 순위")
        
        selected_country = st.selectbox(
            "국가 선택",
            ["전체"] + list(NETFLIX_DATA.keys())
        )
        
        categories = set()
        for country_data in NETFLIX_DATA.values():
            for series_data in country_data["작품"].values():
                categories.add(series_data["카테고리"])
        
        selected_category = st.selectbox(
            "카테고리 선택",
            ["전체"] + sorted(list(categories))
        )
        
        if st.button("📊 순위 조회", use_container_width=True):
            if selected_country == "전체":
                rankings = get_country_rankings(selected_category if selected_category != "전체" else None)
                response = "## 🌍 전세계 작품 순위\n\n"
                for country, series_list in rankings.items():
                    if series_list:
                        response += f"### 🇺🇳 {country}\n\n"
                        for series in series_list:
                            response += f"{series['순위']}. **{series['작품명']}** ({series['카테고리']})\n"
                            response += f"   ⭐ {series['별점']}/5.0 | 💬 \"{series['유명_명대사']}\"\n\n"
                        response += "---\n\n"
            else:
                rankings = get_country_rankings(selected_category if selected_category != "전체" else None)
                if selected_country in rankings and rankings[selected_country]:
                    response = f"## 🇺🇳 {selected_country} 작품 순위\n\n"
                    for series in rankings[selected_country]:
                        response += f"{series['순위']}. **{series['작품명']}** ({series['카테고리']})\n"
                        response += f"   ⭐ {series['별점']}/5.0 | 💬 \"{series['유명_명대사']}\"\n\n"
                else:
                    response = f"{selected_country}의 {selected_category if selected_category != '전체' else ''} 작품을 찾을 수 없어요!"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        
        st.markdown("---")
        st.header("🎬 빠른 조회")
        
        country_list = list(NETFLIX_DATA.keys())
        selected_country_quick = st.selectbox(
            "국가",
            ["국가 선택"] + country_list,
            key="quick_country"
        )
        
        if selected_country_quick != "국가 선택":
            series_list = list(NETFLIX_DATA[selected_country_quick]["작품"].keys())
            selected_series_quick = st.selectbox(
                "작품",
                ["작품 선택"] + series_list,
                key="quick_series"
            )
            
            if selected_series_quick != "작품 선택":
                characters = NETFLIX_DATA[selected_country_quick]["작품"][selected_series_quick]["캐릭터"]
                selected_character_quick = st.selectbox(
                    "캐릭터",
                    ["전체"] + list(characters.keys()),
                    key="quick_character"
                )
                
                if st.button("✨ 정보 조회", use_container_width=True, key="quick_search"):
                    if selected_character_quick == "전체":
                        info = format_character_response(selected_country_quick, selected_series_quick)
                    else:
                        info = format_character_response(selected_country_quick, selected_series_quick, selected_character_quick)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": info
                    })
                    st.rerun()
    
    # 채팅 인터페이스
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("🎬 어떤 작품이나 캐릭터가 궁금하세요? (예: 한국 오징어 게임 성기훈)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        country, series, character = extract_series_and_character(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("정보를 찾고 있어요... 🎬✨"):
                if country and series:
                    character_info = format_character_response(country, series, character)
                    ai_response = get_chat_response(
                        f"사용자가 {country}의 {series}의 {character if character else '캐릭터들'}에 대해 물어봤어요. 다음 정보를 바탕으로 귀엽고 친근하게 추가 설명을 해주세요:\n\n{character_info}",
                        st.session_state.conversation_history,
                        temperature,
                        max_tokens,
                        top_p
                    )
                    response = f"{character_info}\n\n---\n\n{ai_response}"
                else:
                    response = get_chat_response(prompt, st.session_state.conversation_history, temperature, max_tokens, top_p)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
